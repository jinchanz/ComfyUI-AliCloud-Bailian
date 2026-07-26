"""
阿里云百炼 HappyHorse 视频生成统一节点
支持所有 HappyHorse 模型的全部功能：
- 文生视频 (T2V): 纯文本提示词生成视频
- 图生视频 (I2V): 首帧图片 + 文本引导
- 参考生视频 (R2V): 多张参考图 + 文本融合生成
- 视频编辑 (Video Edit): 输入视频 + 参考图 + 编辑指令

API 采用异步调用方式：创建任务 -> 轮询获取结果
"""

import base64
import io
import os
import requests
import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .logging import logger


class HappyHorseVideoGeneration:
    """百炼 HappyHorse 视频生成统一节点

    自动根据输入判断生成模式：
    - 无图片/视频输入 → 文生视频 (T2V)
    - 有首帧图片输入 → 图生视频 (I2V)
    - 有参考图片输入（无视频） → 参考生视频 (R2V)
    - 有视频输入 → 视频编辑 (Video Edit)
    """

    # 模型版本与模式的映射关系
    MODEL_MAP = {
        "1.1": {
            "t2v": "happyhorse-1.1-t2v",
            "i2v": "happyhorse-1.1-i2v",
            "r2v": "happyhorse-1.1-r2v",
            "video_edit": "happyhorse-1.0-video-edit",  # 视频编辑仅1.0
        },
        "1.0": {
            "t2v": "happyhorse-1.0-t2v",
            "i2v": "happyhorse-1.0-i2v",
            "r2v": "happyhorse-1.0-r2v",
            "video_edit": "happyhorse-1.0-video-edit",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一座由硬纸板和瓶盖搭建的微型城市，在夜晚焕发出生机。",
                    "multiline": True
                }),
                "version": (["1.1", "1.0"], {"default": "1.1"}),
            },
            "optional": {
                # --- 首帧图片 (I2V 模式) ---
                "first_frame": ("IMAGE",),
                "first_frame_url": ("STRING", {"default": "", "placeholder": "首帧图片URL (I2V模式)"}),
                # --- 参考图片 (R2V / Video Edit 模式，支持batch多图) ---
                "ref_images": ("IMAGE",),
                "ref_image_urls": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "参考图片URL，多个用换行分隔 (R2V/编辑模式)"
                }),
                # --- 视频输入 (Video Edit 模式) ---
                "video_url": ("STRING", {"default": "", "placeholder": "待编辑视频URL (Video Edit模式)"}),
                # --- 生成参数 ---
                "resolution": (["1080P", "720P"], {"default": "1080P"}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9"], {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                # --- 视频编辑专用参数 ---
                "audio_setting": (["auto", "origin"], {"default": "auto"}),
                # --- 连接配置 ---
                "api_key": ("STRING", {"default": "", "placeholder": "百炼 API Key (sk-xxx)"}),
                "workspace_id": ("STRING", {"default": "", "placeholder": "业务空间ID，留空使用默认域名"}),
                "region": (["cn-beijing", "ap-southeast-1", "us", "eu-central-1"], {"default": "cn-beijing"}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 15, "min": 5, "max": 60, "step": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview_frame", "video_url", "info")
    FUNCTION = "generate"
    CATEGORY = "Malette/Video"

    def generate(self, prompt, version="1.1",
                 first_frame=None, first_frame_url="",
                 ref_images=None, ref_image_urls="",
                 video_url="",
                 resolution="1080P", ratio="16:9", duration=5,
                 watermark=True, seed=-1, audio_setting="auto",
                 api_key="", workspace_id="", region="cn-beijing",
                 timeout=600, poll_interval=15):
        try:
            # 获取 API Key
            api_key = self._resolve_api_key(api_key)

            # 判断生成模式
            mode, mode_desc = self._detect_mode(first_frame, first_frame_url,
                                                 ref_images, ref_image_urls, video_url)

            # 根据模式和版本自动匹配模型
            model = self._resolve_model(version, mode)

            # 构建请求体
            request_body = self._build_request_body(
                model=model, prompt=prompt, mode=mode,
                first_frame=first_frame, first_frame_url=first_frame_url,
                ref_images=ref_images, ref_image_urls=ref_image_urls,
                video_url=video_url,
                resolution=resolution, ratio=ratio, duration=duration,
                watermark=watermark, seed=seed, audio_setting=audio_setting
            )

            # 构建请求 URL
            endpoint = self._build_endpoint(workspace_id, region)

            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "X-DashScope-Async": "enable"
            }

            logger.info(f"[HappyHorse] 开始视频生成 | 模式: {mode_desc} | 模型: {model}")
            logger.info(f"[HappyHorse] 分辨率: {resolution} | 比例: {ratio} | 时长: {duration}s")

            # 步骤1: 创建任务
            response = requests.post(endpoint, headers=headers, json=request_body, timeout=30)

            if response.status_code != 200:
                raise Exception(f"提交任务失败: HTTP {response.status_code} - {response.text}")

            result = response.json()

            if "code" in result:
                raise Exception(f"API 错误 [{result.get('code')}]: {result.get('message', '未知错误')}")

            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                raise Exception(f"响应中未找到 task_id: {json.dumps(result, ensure_ascii=False)[:500]}")

            logger.info(f"[HappyHorse] 任务已提交, task_id: {task_id}")

            # 步骤2: 轮询获取结果
            task_result = self._poll_task_result(task_id, api_key, workspace_id, region, timeout, poll_interval)

            # 处理结果
            output = task_result.get("output", {})
            final_status = output.get("task_status", "")

            if final_status != "SUCCEEDED":
                error_code = output.get("code", "unknown")
                error_message = output.get("message", "任务执行失败")
                raise Exception(f"任务失败 [{error_code}]: {error_message}")

            video_result_url = output.get("video_url", "")
            if not video_result_url:
                raise Exception("任务成功但未返回 video_url")

            logger.info(f"[HappyHorse] 视频生成成功! URL: {video_result_url}")

            # 构建信息字符串
            usage = task_result.get("usage", {})
            info = (
                f"[{mode_desc}] 模型: {model} | 分辨率: {usage.get('SR', resolution)} | "
                f"比例: {usage.get('ratio', ratio)} | "
                f"时长: {usage.get('output_video_duration', duration)}s | "
                f"task_id: {task_id}"
            )

            # 提取视频第一帧作为预览
            preview_frame = self._extract_frame_from_video(video_result_url, timeout)
            if preview_frame is None:
                preview_frame = self._create_placeholder_frame()
                info += " (无法提取预览帧)"

            return (preview_frame, video_result_url, info)

        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            logger.error(f"[HappyHorse] {error_msg}")
            raise Exception(error_msg)

    # ----------------------------------------------------------------
    # 模式检测
    # ----------------------------------------------------------------

    def _detect_mode(self, first_frame, first_frame_url, ref_images, ref_image_urls, video_url):
        """根据输入自动检测生成模式"""
        has_video = bool(video_url and video_url.strip())
        has_first_frame = (first_frame is not None) or bool(first_frame_url and first_frame_url.strip())
        has_ref_images = (ref_images is not None) or bool(ref_image_urls and ref_image_urls.strip())

        # 优先级：视频编辑 > 图生视频 > 参考生视频 > 文生视频
        if has_video:
            return "video_edit", "视频编辑"
        elif has_first_frame:
            return "i2v", "图生视频"
        elif has_ref_images:
            return "r2v", "参考生视频"
        else:
            return "t2v", "文生视频"

    def _resolve_model(self, version, mode):
        """根据版本和模式自动匹配正确的模型名称"""
        version_map = self.MODEL_MAP.get(version)
        if not version_map:
            raise ValueError(f"不支持的模型版本: {version}，可选: 1.1, 1.0")

        model = version_map.get(mode)
        if not model:
            raise ValueError(f"版本 {version} 不支持模式: {mode}")

        logger.info(f"[HappyHorse] 自动匹配模型: 版本={version}, 模式={mode} → {model}")
        return model

    # ----------------------------------------------------------------
    # 构建请求
    # ----------------------------------------------------------------

    def _build_request_body(self, model, prompt, mode,
                            first_frame, first_frame_url,
                            ref_images, ref_image_urls, video_url,
                            resolution, ratio, duration,
                            watermark, seed, audio_setting):
        """根据模式构建请求体"""

        if mode == "t2v":
            # 文生视频: input = {prompt}, parameters = {resolution, ratio, duration, watermark, seed}
            request_body = {
                "model": model,
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "resolution": resolution,
                    "ratio": ratio,
                    "duration": duration,
                    "watermark": watermark,
                }
            }

        elif mode == "i2v":
            # 图生视频: input = {prompt, media: [{type: "first_frame", url}]}
            media = []
            if first_frame is not None:
                base64_data = self._image_tensor_to_base64(first_frame)
                media.append({"type": "first_frame", "url": f"data:image/png;base64,{base64_data}"})
            elif first_frame_url and first_frame_url.strip():
                media.append({"type": "first_frame", "url": first_frame_url.strip()})

            request_body = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "media": media
                },
                "parameters": {
                    "resolution": resolution,
                    "duration": duration,
                    "watermark": watermark,
                    # I2V 不支持 ratio，宽高比跟随首帧图
                }
            }

        elif mode == "r2v":
            # 参考生视频: input = {prompt, media: [{type: "reference_image", url}, ...]}
            media = self._build_reference_image_media(ref_images, ref_image_urls)

            request_body = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "media": media
                },
                "parameters": {
                    "resolution": resolution,
                    "ratio": ratio,
                    "duration": duration,
                    "watermark": watermark,
                }
            }

        elif mode == "video_edit":
            # 视频编辑: input = {prompt, media: [{type: "video", url}, {type: "reference_image", url}, ...]}
            media = [{"type": "video", "url": video_url.strip()}]

            # 添加参考图片（可选，0-5张）
            ref_media = self._build_reference_image_media(ref_images, ref_image_urls)
            media.extend(ref_media)

            request_body = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "media": media
                },
                "parameters": {
                    "resolution": resolution,
                    "watermark": watermark,
                    "audio_setting": audio_setting,
                    # 视频编辑不支持 ratio 和 duration
                }
            }
        else:
            raise ValueError(f"未知的生成模式: {mode}")

        # 添加 seed（所有模式通用）
        if seed >= 0:
            request_body["parameters"]["seed"] = seed

        return request_body

    def _build_reference_image_media(self, ref_images, ref_image_urls):
        """构建参考图片 media 列表"""
        media = []

        # 从 IMAGE tensor 构建 (支持 batch 多图)
        if ref_images is not None and isinstance(ref_images, torch.Tensor):
            batch_size = ref_images.shape[0] if len(ref_images.shape) == 4 else 1
            for i in range(batch_size):
                single_image = ref_images[i] if len(ref_images.shape) == 4 else ref_images
                base64_data = self._image_tensor_to_base64_single(single_image)
                media.append({
                    "type": "reference_image",
                    "url": f"data:image/png;base64,{base64_data}"
                })

        # 从 URL 列表构建
        if ref_image_urls and ref_image_urls.strip():
            for url in self._parse_urls(ref_image_urls):
                media.append({
                    "type": "reference_image",
                    "url": url
                })

        return media

    # ----------------------------------------------------------------
    # 网络与轮询
    # ----------------------------------------------------------------

    def _resolve_api_key(self, api_key):
        """获取 API Key"""
        if api_key and api_key.strip():
            return api_key.strip()
        env_key = os.getenv("DASHSCOPE_API_KEY")
        if env_key:
            return env_key
        raise Exception("未设置 API Key，请在节点参数中输入或设置环境变量 DASHSCOPE_API_KEY")

    def _build_endpoint(self, workspace_id, region):
        """根据地域和 workspace_id 构建请求端点"""
        path = "/api/v1/services/aigc/video-generation/video-synthesis"

        if region == "us":
            return f"https://dashscope-us.aliyuncs.com{path}"

        if workspace_id and workspace_id.strip():
            ws_id = workspace_id.strip()
            if region == "cn-beijing":
                return f"https://{ws_id}.cn-beijing.maas.aliyuncs.com{path}"
            elif region == "ap-southeast-1":
                return f"https://{ws_id}.ap-southeast-1.maas.aliyuncs.com{path}"
            elif region == "eu-central-1":
                return f"https://{ws_id}.eu-central-1.maas.aliyuncs.com{path}"

        return f"https://dashscope.aliyuncs.com{path}"

    def _build_task_url(self, task_id, workspace_id, region):
        """构建任务查询 URL"""
        if region == "us":
            return f"https://dashscope-us.aliyuncs.com/api/v1/tasks/{task_id}"

        if workspace_id and workspace_id.strip():
            ws_id = workspace_id.strip()
            if region == "cn-beijing":
                return f"https://{ws_id}.cn-beijing.maas.aliyuncs.com/api/v1/tasks/{task_id}"
            elif region == "ap-southeast-1":
                return f"https://{ws_id}.ap-southeast-1.maas.aliyuncs.com/api/v1/tasks/{task_id}"
            elif region == "eu-central-1":
                return f"https://{ws_id}.eu-central-1.maas.aliyuncs.com/api/v1/tasks/{task_id}"

        return f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

    def _poll_task_result(self, task_id, api_key, workspace_id, region, timeout, poll_interval):
        """轮询任务结果"""
        task_url = self._build_task_url(task_id, workspace_id, region)
        headers = {"Authorization": f"Bearer {api_key}"}

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"轮询超时: 已等待 {int(elapsed)}s，超过 {timeout}s 限制")

            try:
                logger.info(f"[HappyHorse] 轮询任务状态: {task_id} ({int(elapsed)}s)")

                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()
                task_status = result.get("output", {}).get("task_status", "")

                logger.info(f"[HappyHorse] 任务状态: {task_status}")

                if task_status == "SUCCEEDED":
                    return result
                elif task_status == "FAILED":
                    return result
                elif task_status == "CANCELED":
                    raise Exception("任务已被取消")
                elif task_status == "UNKNOWN":
                    raise Exception("任务不存在或已过期 (UNKNOWN)")
                elif task_status in ["PENDING", "RUNNING"]:
                    time.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"[HappyHorse] 未知状态: {task_status}，继续轮询...")
                    time.sleep(poll_interval)
                    continue

            except requests.exceptions.RequestException as e:
                if time.time() - start_time > timeout:
                    raise Exception(f"轮询超时，最后网络错误: {str(e)}")
                logger.warning(f"[HappyHorse] 网络错误: {str(e)}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

    # ----------------------------------------------------------------
    # 图像工具方法
    # ----------------------------------------------------------------

    def _image_tensor_to_base64(self, image):
        """将 ComfyUI IMAGE tensor 转换为 base64 (取第一张)"""
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
        return self._image_tensor_to_base64_single(image)

    def _image_tensor_to_base64_single(self, image_tensor):
        """将单张 (H,W,C) tensor 转换为 base64"""
        image_array = image_tensor.cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_urls(self, urls_text):
        """解析多行/逗号分隔的 URL 列表"""
        if not urls_text or not urls_text.strip():
            return []
        urls = []
        for line in urls_text.replace(",", "\n").split("\n"):
            url = line.strip()
            if url:
                urls.append(url)
        return urls

    def _extract_frame_from_video(self, video_url, timeout):
        """使用 ffmpeg 从视频 URL 提取第一帧作为预览"""
        # 提前检测 ffmpeg 是否存在，避免抛出 FileNotFoundError
        if shutil.which("ffmpeg") is None:
            logger.info("[HappyHorse] 未检测到 ffmpeg，跳过预览帧提取（不影响视频生成）")
            return None
        try:
            temp_dir = Path(tempfile.gettempdir())
            timestamp = int(time.time())
            video_file = temp_dir / f"happyhorse_video_{timestamp}.mp4"
            frame_file = temp_dir / f"happyhorse_frame_{timestamp}.png"

            try:
                logger.info("[HappyHorse] 下载视频用于帧提取...")
                response = requests.get(video_url, timeout=timeout)
                if response.status_code != 200:
                    logger.warning(f"[HappyHorse] 下载视频失败: HTTP {response.status_code}")
                    return None

                with open(video_file, "wb") as f:
                    f.write(response.content)

                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_file),
                     "-vframes", "1", "-q:v", "2", str(frame_file)],
                    capture_output=True, timeout=30,
                )

                if frame_file.exists():
                    pil_image = Image.open(frame_file)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    return torch.from_numpy(image_array).unsqueeze(0)
            finally:
                if video_file.exists():
                    video_file.unlink(missing_ok=True)
                if frame_file.exists():
                    frame_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"[HappyHorse] 提取视频帧失败: {str(e)}")

        return None

    def _create_placeholder_frame(self):
        """创建占位预览图"""
        placeholder = np.full((720, 1280, 3), 0.2, dtype=np.float32)
        return torch.from_numpy(placeholder).unsqueeze(0)


class UnifyVideoGeneration:
    """中台统一视频生成节点

    支持多个视频生成服务的统一调用：
    - Kling（可灵）: v1 / v1.5 / v2 / v3
    - MiniMax（海螺）: Hailuo-2.3
    - Wan（万相）: wan2.7 系列
    - Seedance: seedance-1-0 / 2-0

    自动根据输入判断生成模式 (T2V/I2V/R2V/V2V)，自动匹配正确的模型名称。
    用户只需选择模型源和版本，无需手动指定模式和模型名。
    """

    DEFAULT_BASE_URL = "/api/video/unify"

    # 模型源 + 版本 + 模式 → 模型名称 自动匹配表
    # 对于 kling/minimax/seedance，所有模式使用同一个模型名
    # 对于 wan，不同模式对应不同的模型名
    MODEL_MAP = {
        "kling": {
            "kling-v3":   {"T2V": "kling-v3", "I2V": "kling-v3", "R2V": "kling-v3"},
            "kling-v3-omni":   {"T2V": "kling-v3-omni", "I2V": "kling-v3-omni", "R2V": "kling-v3-omni"},
            "kling-3.0-turbo": {"T2V": "kling-3.0-turbo", "I2V": "kling-3.0-turbo", "R2V": "kling-3.0-turbo"},
            "kling-v2-5-turbo":   {"T2V": "kling-v2-5-turbo", "I2V": "kling-v2-5-turbo", "R2V": "kling-v2-5-turbo"},
        },
        "minimax": {
            "2.3": {"T2V": "MiniMax-Hailuo-2.3", "I2V": "MiniMax-Hailuo-2.3", "R2V": "MiniMax-Hailuo-2.3", "V2V": "MiniMax-Hailuo-2.3"},
        },
        "wan": {
            "2.7": {"T2V": "wan2.7-t2v", "I2V": "wan2.7-i2v", "R2V": "wan2.7-r2v", "V2V": "wan2.7-videoedit"},
        },
        "seedream": {
            "2.0": {"T2V": "doubao-seedance-2-0-260128", "I2V": "doubao-seedance-2-0-260128", "R2V": "doubao-seedance-2-0-260128", "V2V": "doubao-seedance-2-0-260128"},
            "mini": {"T2V": "doubao-seedance-2-0-mini-260615", "I2V": "doubao-seedance-2-0-mini-260615", "R2V": "doubao-seedance-2-0-mini-260615", "V2V": "doubao-seedance-2-0-mini-260615"},
        },
    }

    # 各模型源的默认版本
    DEFAULT_VERSIONS = {
        "kling": "v3",
        "minimax": "2.3",
        "wan": "2.7",
        "seedream": "2.0",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一只可爱的小猫在草地上奔跑",
                    "multiline": True
                }),
                "model_source": (["kling", "minimax", "wan", "seedream"],),
                "version": ("STRING", {
                    "default": "kling-v3",
                    "placeholder": "版本: kling(v1/v1.5/v2/v3) wan(2.7) minimax(2.3) seedream(1.0/2.0)"
                }),
                "base_url": ("STRING", {
                    "default": "",
                    "placeholder": "中台API基础地址，如 https://your-domain.com/api/video/unify"
                }),
            },
            "optional": {
                # --- 提示词参数 ---
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "负向提示词"}),
                "prompt_extend": ("BOOLEAN", {"default": False}),
                # --- 视频参数 ---
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "21:9"], {"default": "16:9"}),
                "resolution": (["1080p", "720p", "480p"], {"default": "1080p"}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 15, "step": 1}),
                "seed": ("STRING", {"default": "", "placeholder": "随机种子，留空为随机"}),
                "watermark": ("BOOLEAN", {"default": False}),
                # --- 音频参数 ---
                "generate_audio": ("BOOLEAN", {"default": False}),
                "audio_setting": (["auto", "origin"], {"default": "auto"}),
                # --- Kling 专用参数 ---
                "mode": (["std", "pro"], {"default": "std"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                # --- 媒体输入 (I2V/R2V/V2V) ---
                "first_frame": ("IMAGE",),
                "first_frame_url": ("STRING", {"default": "", "placeholder": "首帧图片URL"}),
                "last_frame": ("IMAGE",),
                "last_frame_url": ("STRING", {"default": "", "placeholder": "尾帧图片URL"}),
                "ref_images": ("IMAGE",),
                "ref_image_urls": ("STRING", {"default": "", "multiline": True, "placeholder": "参考图片URL，多个换行分隔"}),
                "video_url": ("STRING", {"default": "", "placeholder": "视频URL (V2V模式)"}),
                "audio_url": ("STRING", {"default": "", "placeholder": "驱动音频URL"}),
                # --- 多镜头 ---
                "multi_shot": ("BOOLEAN", {"default": False}),
                "shot_type": (["customize", "intelligence"], {"default": "intelligence"}),
                "multi_prompt_json": ("STRING", {"default": "", "multiline": True, "placeholder": '多镜头JSON，如 [{"content":"...","duration":5}]'}),
                # --- 调用配置 ---
                "token": ("STRING", {"default": "", "placeholder": "Bearer Token"}),
                "source": ("STRING", {"default": "", "placeholder": "业务方来源标识"}),
                "operator": ("STRING", {"default": "", "placeholder": "操作者ID"}),
                "biz_id": ("STRING", {"default": "", "placeholder": "业务ID（幂等）"}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 8, "min": 3, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview_frame", "video_url", "info")
    FUNCTION = "generate"
    CATEGORY = "Malette/Video"

    def generate(self, prompt, model_source, version="", base_url="",
                 negative_prompt="", prompt_extend=False,
                 ratio="16:9", resolution="1080p", duration=5,
                 seed="", watermark=False,
                 generate_audio=False, audio_setting="auto",
                 mode="std", cfg_scale=0.5,
                 first_frame=None, first_frame_url="",
                 last_frame=None, last_frame_url="",
                 ref_images=None, ref_image_urls="",
                 video_url="", audio_url="",
                 multi_shot=False, shot_type="intelligence", multi_prompt_json="",
                 token="", source="", operator="", biz_id="",
                 timeout=600, poll_interval=8):
        try:
            # 自动检测生成模式
            generate_type = self._detect_generate_type(
                first_frame, first_frame_url, last_frame, last_frame_url,
                ref_images, ref_image_urls, video_url
            )

            # 自动匹配模型名称
            model_name = self._resolve_model_name(model_source, version, generate_type)

            logger.info(f"[UnifyVideo] 自动匹配: 模型源={model_source}, 版本={version}, "
                        f"模式={generate_type} → {model_name}")

            # 构建请求体
            request_body = self._build_request_body(
                prompt=prompt, model_source=model_source, model_name=model_name,
                generate_type=generate_type, negative_prompt=negative_prompt,
                prompt_extend=prompt_extend, ratio=ratio, resolution=resolution,
                duration=duration, seed=seed, watermark=watermark,
                generate_audio=generate_audio, audio_setting=audio_setting,
                mode=mode, cfg_scale=cfg_scale,
                first_frame=first_frame, first_frame_url=first_frame_url,
                last_frame=last_frame, last_frame_url=last_frame_url,
                ref_images=ref_images, ref_image_urls=ref_image_urls,
                video_url=video_url, audio_url=audio_url,
                multi_shot=multi_shot, shot_type=shot_type,
                multi_prompt_json=multi_prompt_json,
                source=source, operator=operator, biz_id=biz_id
            )

            # 构建请求头
            headers = {
                "Content-Type": "application/json",
            }
            if token and token.strip():
                headers["Authorization"] = f"Bearer {token.strip()}"

            logger.info(f"[UnifyVideo] 提交任务 | {model_source}/{model_name} | 模式: {generate_type}")
            logger.info(f"[UnifyVideo] 请求体: {self._safe_body_for_log(request_body)}")

            # 步骤1: 提交任务
            api_base = base_url.strip() if base_url and base_url.strip() else self.DEFAULT_BASE_URL
            generate_url = f"{api_base}/generate"
            response = requests.post(generate_url, headers=headers, json=request_body, timeout=30)

            if response.status_code != 200:
                raise Exception(f"提交任务失败: HTTP {response.status_code} - {response.text}")

            result = response.json()
            if not result.get("success"):
                raise Exception(f"提交任务失败: [{result.get('code')}] {result.get('message')}")

            task_id = result.get("data")
            if not task_id:
                raise Exception(f"响应中未找到 taskId: {json.dumps(result, ensure_ascii=False)[:500]}")

            logger.info(f"[UnifyVideo] 任务已提交, taskId: {task_id}")

            # 步骤2: 轮询查询结果
            self._api_base = api_base
            video_result = self._poll_task_result(task_id, model_source, headers, timeout, poll_interval)

            # 提取结果
            status = video_result.get("status", "")
            if status != "DONE":
                error_code = video_result.get("errorCode", "unknown")
                error_message = video_result.get("errorMessage", "任务执行失败")
                raise Exception(f"任务失败 [{error_code}]: {error_message}")

            video_result_url = video_result.get("videoUrl", "")
            if not video_result_url:
                raise Exception("任务成功但未返回 videoUrl")

            logger.info(f"[UnifyVideo] 视频生成成功! URL: {video_result_url}")

            # 构建信息
            width = video_result.get("width", "?")
            height = video_result.get("height", "?")
            vid_duration = video_result.get("duration", duration)
            info = (
                f"[{generate_type}] 模型: {model_source}/{model_name} | "
                f"分辨率: {width}x{height} | 时长: {vid_duration}s | "
                f"taskId: {task_id}"
            )

            # 提取视频第一帧作为预览
            preview_frame = self._extract_frame_from_video(video_result_url, timeout)
            if preview_frame is None:
                preview_frame = self._create_placeholder_frame()
                info += " (无法提取预览帧)"

            return (preview_frame, video_result_url, info)

        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            logger.error(f"[UnifyVideo] {error_msg}")
            raise Exception(error_msg)

    # ----------------------------------------------------------------
    # 模式检测与模型匹配
    # ----------------------------------------------------------------

    def _detect_generate_type(self, first_frame, first_frame_url,
                              last_frame, last_frame_url,
                              ref_images, ref_image_urls, video_url):
        """根据输入自动检测生成模式"""
        has_video = bool(video_url and video_url.strip())
        has_first_frame = (first_frame is not None) or bool(first_frame_url and first_frame_url.strip())
        has_last_frame = (last_frame is not None) or bool(last_frame_url and last_frame_url.strip())
        has_ref_images = (ref_images is not None) or bool(ref_image_urls and ref_image_urls.strip())

        # 优先级: V2V > I2V > R2V > T2V
        if has_video:
            return "V2V"
        elif has_first_frame or has_last_frame:
            return "I2V"
        elif has_ref_images:
            return "R2V"
        else:
            return "T2V"

    def _resolve_model_name(self, model_source, version, generate_type):
        """根据模型源、版本和模式自动匹配模型名称"""
        source_map = self.MODEL_MAP.get(model_source)
        if not source_map:
            raise ValueError(f"不支持的模型源: {model_source}，可选: kling, minimax, wan, seedance")

        # 如果版本为空，使用默认版本
        if not version or not version.strip():
            version = self.DEFAULT_VERSIONS.get(model_source, "")

        version = version.strip()
        version_map = source_map.get(version)
        if not version_map:
            available = list(source_map.keys())
            raise ValueError(f"{model_source} 不支持版本 '{version}'，可选: {available}")

        model_name = version_map.get(generate_type)
        if not model_name:
            raise ValueError(f"{model_source}/{version} 不支持模式: {generate_type}")

        return model_name

    # ----------------------------------------------------------------
    # 构建请求
    # ----------------------------------------------------------------

    def _build_request_body(self, prompt, model_source, model_name, generate_type,
                            negative_prompt, prompt_extend, ratio, resolution,
                            duration, seed, watermark, generate_audio, audio_setting,
                            mode, cfg_scale, first_frame, first_frame_url,
                            last_frame, last_frame_url, ref_images, ref_image_urls,
                            video_url, audio_url, multi_shot, shot_type,
                            multi_prompt_json, source, operator, biz_id):
        """构建统一请求体"""
        body = {
            "modelSource": model_source,
            "modelName": model_name,
            "generateType": generate_type,
            "prompt": prompt,
            "resolution": resolution,
            "duration": duration,
            "waterMark": watermark,
            "generateAudio": generate_audio,
            "promptExtend": prompt_extend,
        }

        # ratio 仅在 T2V / R2V 模式下传递；
        # I2V(跟随首帧图) 和 V2V(跟随源视频) 的输出宽高比由输入决定，不传 ratio
        if generate_type in ("T2V", "R2V"):
            body["ratio"] = ratio

        # 可选字段
        if negative_prompt and negative_prompt.strip():
            body["negativePrompt"] = negative_prompt.strip()
        if seed and seed.strip():
            body["seed"] = seed.strip()
        if source and source.strip():
            body["source"] = source.strip()
        if operator and operator.strip():
            body["operator"] = operator.strip()
        if biz_id and biz_id.strip():
            body["bizId"] = biz_id.strip()

        # Kling 专用参数
        if model_source == "kling":
            body["mode"] = mode
            body["cfgScale"] = str(cfg_scale)

        # Wan 专用参数
        if model_source == "wan":
            body["audioSetting"] = audio_setting

        # 多镜头
        if multi_shot:
            body["multiShot"] = True
            body["shotType"] = shot_type
            if multi_prompt_json and multi_prompt_json.strip():
                try:
                    body["multiPromptList"] = json.loads(multi_prompt_json.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"[UnifyVideo] 多镜头JSON解析失败: {e}")

        # 媒体列表
        media_list = self._build_media_list(
            first_frame=first_frame, first_frame_url=first_frame_url,
            last_frame=last_frame, last_frame_url=last_frame_url,
            ref_images=ref_images, ref_image_urls=ref_image_urls,
            video_url=video_url, audio_url=audio_url
        )
        if media_list:
            body["mediaList"] = media_list

        return body

    def _build_media_list(self, first_frame, first_frame_url,
                          last_frame, last_frame_url,
                          ref_images, ref_image_urls,
                          video_url, audio_url):
        """构建媒体列表"""
        media_list = []

        # 首帧
        if first_frame is not None:
            base64_data = self._image_tensor_to_base64(first_frame)
            media_list.append({"content": f"data:image/png;base64,{base64_data}", "role": "firstFrame"})
        elif first_frame_url and first_frame_url.strip():
            media_list.append({"content": first_frame_url.strip(), "role": "firstFrame"})

        # 尾帧
        if last_frame is not None:
            base64_data = self._image_tensor_to_base64(last_frame)
            media_list.append({"content": f"data:image/png;base64,{base64_data}", "role": "lastFrame"})
        elif last_frame_url and last_frame_url.strip():
            media_list.append({"content": last_frame_url.strip(), "role": "lastFrame"})

        # 参考图片 (batch IMAGE tensor)
        if ref_images is not None and isinstance(ref_images, torch.Tensor):
            batch_size = ref_images.shape[0] if len(ref_images.shape) == 4 else 1
            for i in range(batch_size):
                single_image = ref_images[i] if len(ref_images.shape) == 4 else ref_images
                base64_data = self._image_tensor_to_base64_single(single_image)
                media_list.append({"content": f"data:image/png;base64,{base64_data}", "role": "referenceImage"})

        # 参考图片 URL
        if ref_image_urls and ref_image_urls.strip():
            for url in self._parse_urls(ref_image_urls):
                media_list.append({"content": url, "role": "referenceImage"})

        # 视频 (V2V模式)
        if video_url and video_url.strip():
            media_list.append({"content": video_url.strip(), "role": "targetVideo"})

        # 参考音频
        if audio_url and audio_url.strip():
            media_list.append({"content": audio_url.strip(), "role": "referenceVoice"})

        return media_list

    def _safe_body_for_log(self, body):
        """截断 base64 媒体内容，生成可读的日志字符串"""
        import copy
        safe = copy.deepcopy(body)
        medias = safe.get("mediaList")
        if isinstance(medias, list):
            for item in medias:
                content = item.get("content", "")
                if isinstance(content, str) and content.startswith("data:"):
                    item["content"] = content[:50] + f"...(base64共{len(content)}字符)"
        return json.dumps(safe, ensure_ascii=False)

    # ----------------------------------------------------------------
    # 轮询
    # ----------------------------------------------------------------

    def _poll_task_result(self, task_id, model_source, headers, timeout, poll_interval):
        """轮询任务结果"""
        query_url = f"{self._api_base}/query"
        params = {"taskId": task_id, "modelSource": model_source}

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"轮询超时: 已等待 {int(elapsed)}s，超过 {timeout}s 限制")

            try:
                logger.info(f"[UnifyVideo] 轮询任务: {task_id} ({int(elapsed)}s)")

                response = requests.get(query_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()

                result = response.json()

                if not result.get("success"):
                    raise Exception(f"查询失败: [{result.get('code')}] {result.get('message')}")

                data = result.get("data", {})
                status = data.get("status", "")

                logger.info(f"[UnifyVideo] 任务状态: {status}")

                if status == "DONE":
                    return data
                elif status == "ERROR":
                    return data
                elif status == "PROCESSING" or status == "RUNNING":
                    time.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"[UnifyVideo] 未知状态: {status}，继续轮询...")
                    time.sleep(poll_interval)
                    continue

            except requests.exceptions.RequestException as e:
                if time.time() - start_time > timeout:
                    raise Exception(f"轮询超时，最后网络错误: {str(e)}")
                logger.warning(f"[UnifyVideo] 网络错误: {str(e)}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

    # ----------------------------------------------------------------
    # 图像工具方法 (复用 HappyHorse 的逻辑)
    # ----------------------------------------------------------------

    def _image_tensor_to_base64(self, image):
        """将 ComfyUI IMAGE tensor 转换为 base64 (取第一张)"""
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
        return self._image_tensor_to_base64_single(image)

    def _image_tensor_to_base64_single(self, image_tensor):
        """将单张 (H,W,C) tensor 转换为 base64"""
        image_array = image_tensor.cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_urls(self, urls_text):
        """解析多行/逗号分隔的 URL 列表"""
        if not urls_text or not urls_text.strip():
            return []
        urls = []
        for line in urls_text.replace(",", "\n").split("\n"):
            url = line.strip()
            if url:
                urls.append(url)
        return urls

    def _extract_frame_from_video(self, video_url, timeout):
        """使用 ffmpeg 从视频 URL 提取第一帧作为预览"""
        # 提前检测 ffmpeg 是否存在，避免抛出 FileNotFoundError
        if shutil.which("ffmpeg") is None:
            logger.info("[UnifyVideo] 未检测到 ffmpeg，跳过预览帧提取（不影响视频生成）")
            return None
        try:
            temp_dir = Path(tempfile.gettempdir())
            timestamp = int(time.time())
            video_file = temp_dir / f"unify_video_{timestamp}.mp4"
            frame_file = temp_dir / f"unify_frame_{timestamp}.png"

            try:
                response = requests.get(video_url, timeout=timeout)
                if response.status_code != 200:
                    return None

                with open(video_file, "wb") as f:
                    f.write(response.content)

                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_file),
                     "-vframes", "1", "-q:v", "2", str(frame_file)],
                    capture_output=True, timeout=30,
                )

                if frame_file.exists():
                    pil_image = Image.open(frame_file)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    return torch.from_numpy(image_array).unsqueeze(0)
            finally:
                if video_file.exists():
                    video_file.unlink(missing_ok=True)
                if frame_file.exists():
                    frame_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"[UnifyVideo] 提取视频帧失败: {str(e)}")

        return None

    def _create_placeholder_frame(self):
        """创建占位预览图"""
        placeholder = np.full((720, 1280, 3), 0.2, dtype=np.float32)
        return torch.from_numpy(placeholder).unsqueeze(0)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "HappyHorseVideoGeneration": HappyHorseVideoGeneration,
    "UnifyVideoGeneration": UnifyVideoGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HappyHorseVideoGeneration": "HappyHorse 视频生成 (Video Generation)",
    "UnifyVideoGeneration": "中台统一视频生成 (Unify Video Generation)",
}
