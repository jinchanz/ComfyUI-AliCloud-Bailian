"""阿里云百炼 通义万相 视频生成节点（直连百炼，不经中台）

依据万相 2.7 官方 API：
- 文生视频  wan2.7-t2v ：input{prompt, negative_prompt, audio_url} + parameters{resolution, ratio, ...}
- 图生视频  wan2.7-i2v ：input{prompt, negative_prompt, media[]} + parameters{resolution, ...}
  素材组合决定具体任务（首帧生视频 / 首尾帧生视频 / 视频续写），合法组合仅：
    first_frame
    first_frame + driving_audio
    first_frame + last_frame
    first_frame + last_frame + driving_audio
    first_clip
    first_clip + last_frame

节点按输入自动判断任务类型并匹配模型，无需手动填模型名；
模型清单来自 model_registry 的 dashscope 直连通道（模型源 wan），
与中台统一节点的清单完全独立（中台侧万相归在 modelSource=dashscope 下，模型名可能不同）。
"""

import json

from .dashscope_video import DashScopeVideoMixin
from .logging import logger
from .model_registry import registry, CHANNEL_DASHSCOPE

WAN_SOURCE = "wan"


class WanVideoGeneration(DashScopeVideoMixin):
    """百炼通义万相视频生成节点（直连版）

    自动根据输入判断任务类型：
    - 无任何素材 → 文生视频 (T2V)
    - 有首帧（可选尾帧/音频） → 图生视频 (I2V)
    - 有首段视频（可选尾帧） → 视频续写 (VIDEOEDIT)
    """

    LOG_PREFIX = "Wan"

    @classmethod
    def INPUT_TYPES(cls):
        versions = registry.versions(CHANNEL_DASHSCOPE, WAN_SOURCE) or ["2.7"]
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "写实风格，一只小黑猫好奇地仰望天空，镜头从平视角度逐渐升高。",
                    "multiline": True
                }),
                "version": (versions, {"default": registry.default_version(CHANNEL_DASHSCOPE, WAN_SOURCE)}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "反向提示词，不超过500字符"
                }),
                # --- 首帧 (图生视频) ---
                "first_frame": ("IMAGE",),
                "first_frame_url": ("STRING", {"default": "", "placeholder": "首帧图片URL，支持 http(s)"}),
                # --- 尾帧 (首尾帧生视频 / 视频续写) ---
                "last_frame": ("IMAGE",),
                "last_frame_url": ("STRING", {"default": "", "placeholder": "尾帧图片URL"}),
                # --- 首段视频 (视频续写，2~10s，仅公网URL) ---
                "first_clip_url": ("STRING", {"default": "", "placeholder": "首段视频URL (视频续写)，2~10秒"}),
                # --- 驱动音频 (2~30s，仅公网URL；留空则模型自动配乐) ---
                "audio_url": ("STRING", {"default": "", "placeholder": "驱动音频URL (wav/mp3, 2~30秒)"}),
                # --- 生成参数 ---
                "resolution": (["1080P", "720P"], {"default": "1080P"}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 15, "step": 1}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                # --- 连接配置 ---
                "model_override": ("STRING", {"default": "", "placeholder": "直接指定模型名，填了则忽略 version"}),
                "api_key": ("STRING", {"default": "", "placeholder": "百炼 API Key (sk-xxx)"}),
                "workspace_id": ("STRING", {"default": "", "placeholder": "业务空间ID，留空使用公共域名"}),
                "region": (["cn-beijing", "ap-southeast-1"], {"default": "cn-beijing"}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 15, "min": 5, "max": 60, "step": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview_frame", "video_url", "info")
    FUNCTION = "generate"
    CATEGORY = "Malette/Video"

    def generate(self, prompt, version="2.7",
                 negative_prompt="",
                 first_frame=None, first_frame_url="",
                 last_frame=None, last_frame_url="",
                 first_clip_url="", audio_url="",
                 resolution="1080P", ratio="16:9", duration=5,
                 prompt_extend=True, watermark=False, seed=-1,
                 model_override="", api_key="", workspace_id="",
                 region="cn-beijing", timeout=900, poll_interval=15):
        try:
            api_key = self._resolve_api_key(api_key)

            # 素材：图片支持 base64，音频/视频必须公网URL
            first_frame_content = self._image_to_url(first_frame, first_frame_url)
            last_frame_content = self._image_to_url(last_frame, last_frame_url)
            first_clip_content = self._require_public_url("首段视频URL", first_clip_url)
            audio_content = self._require_public_url("驱动音频URL", audio_url)

            # 自动判断任务类型并校验素材组合
            mode, mode_desc = self._detect_mode(
                first_frame_content, last_frame_content, first_clip_content, audio_content
            )

            # 自动匹配模型
            model = self._resolve_model(version, mode, model_override)

            request_body = self._build_request_body(
                model=model, mode=mode, prompt=prompt, negative_prompt=negative_prompt,
                first_frame=first_frame_content, last_frame=last_frame_content,
                first_clip=first_clip_content, audio=audio_content,
                resolution=resolution, ratio=ratio, duration=duration,
                prompt_extend=prompt_extend, watermark=watermark, seed=seed
            )

            logger.info(f"[Wan] 开始视频生成 | 任务: {mode_desc} | 模型: {model}")
            logger.info(f"[Wan] 分辨率: {resolution} | 时长: {duration}s | "
                        f"比例: {ratio if mode == 'T2V' else '跟随输入素材'}")
            logger.info(f"[Wan] 请求体: {self._safe_body_for_log(request_body)}")

            # 步骤1: 创建任务
            task_id = self._submit_task(request_body, api_key, workspace_id, region)

            # 步骤2: 轮询结果
            task_result = self._poll_task_result(
                task_id, api_key, workspace_id, region, timeout, poll_interval
            )

            output = task_result.get("output", {})
            if output.get("task_status") != "SUCCEEDED":
                error_code = output.get("code", "unknown")
                error_message = output.get("message", "任务执行失败")
                raise Exception(f"任务失败 [{error_code}]: {error_message}")

            video_result_url = output.get("video_url", "")
            if not video_result_url:
                raise Exception("任务成功但未返回 video_url")

            logger.info(f"[Wan] 视频生成成功! URL: {video_result_url}")

            usage = task_result.get("usage", {})
            info = (
                f"[{mode_desc}] 模型: {model} | 分辨率: {usage.get('SR', resolution)} | "
                f"时长: {usage.get('output_video_duration', duration)}s | "
                f"task_id: {task_id}"
            )

            preview_frame = self._extract_frame_from_video(video_result_url, timeout)
            if preview_frame is None:
                preview_frame = self._create_placeholder_frame()
                info += " (无法提取预览帧)"

            return (preview_frame, video_result_url, info)

        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            logger.error(f"[Wan] {error_msg}")
            raise Exception(error_msg)

    # ----------------------------------------------------------------
    # 任务类型判断与模型匹配
    # ----------------------------------------------------------------

    def _detect_mode(self, first_frame, last_frame, first_clip, audio):
        """
        根据素材组合判断任务类型，同时拦截官方文档中的非法组合，
        避免提交后才被服务端以 InvalidParameter 拒绝。
        """
        if first_clip and first_frame:
            raise ValueError("首段视频与首帧不能同时输入：视频续写只支持 首段视频 或 首段视频+尾帧")
        if first_clip and audio:
            raise ValueError("视频续写不支持驱动音频，请移除音频URL或改用首帧生视频")
        if first_clip:
            return "VIDEOEDIT", "视频续写"

        if first_frame:
            return "I2V", "首尾帧生视频" if last_frame else "首帧生视频"

        if last_frame:
            raise ValueError("仅传尾帧不是合法组合，尾帧必须与首帧或首段视频搭配使用")

        return "T2V", "文生视频"

    def _resolve_model(self, version, mode, model_override=""):
        """匹配模型名：填了 model_override 就直接用，否则查模型清单"""
        if model_override and model_override.strip():
            model = model_override.strip()
            logger.info(f"[Wan] 使用手动指定的模型: {model}")
            return model

        model = registry.resolve(CHANNEL_DASHSCOPE, WAN_SOURCE, version, mode)
        logger.info(f"[Wan] 自动匹配模型: 版本={version}, 任务={mode} → {model}")
        return model

    # ----------------------------------------------------------------
    # 构建请求
    # ----------------------------------------------------------------

    def _build_request_body(self, model, mode, prompt, negative_prompt,
                            first_frame, last_frame, first_clip, audio,
                            resolution, ratio, duration,
                            prompt_extend, watermark, seed):
        """按任务类型构建请求体"""
        input_body = {"prompt": prompt}
        if negative_prompt and negative_prompt.strip():
            input_body["negative_prompt"] = negative_prompt.strip()

        parameters = {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        if mode == "T2V":
            # 文生视频的音频走 input.audio_url，且宽高比由 ratio 指定
            if audio:
                input_body["audio_url"] = audio
            parameters["ratio"] = ratio
        else:
            # 图生视频 / 视频续写的音频走 media 里的 driving_audio，
            # 宽高比跟随首帧或首段视频，不传 ratio
            media = []
            if first_clip:
                media.append({"type": "first_clip", "url": first_clip})
            if first_frame:
                media.append({"type": "first_frame", "url": first_frame})
            if last_frame:
                media.append({"type": "last_frame", "url": last_frame})
            if audio:
                media.append({"type": "driving_audio", "url": audio})
            input_body["media"] = media

        if seed >= 0:
            parameters["seed"] = seed

        return {"model": model, "input": input_body, "parameters": parameters}

    def _safe_body_for_log(self, body):
        """截断 base64 媒体内容，生成可读的日志字符串"""
        import copy
        safe = copy.deepcopy(body)
        input_body = safe.get("input", {})
        for item in input_body.get("media", []) or []:
            url = item.get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                item["url"] = url[:50] + f"...(base64共{len(url)}字符)"
        return json.dumps(safe, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {
    "WanVideoGeneration": WanVideoGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoGeneration": "通义万相 视频生成 (Wan Video Generation)",
}
