"""阿里云百炼（DashScope）视频生成节点的公共能力

HappyHorse / 万相 等模型都走同一套异步接口：
POST /api/v1/services/aigc/video-generation/video-synthesis  创建任务
GET  /api/v1/tasks/{task_id}                                 轮询结果

本模块把 API Key 解析、地域端点拼接、轮询、图像转 base64、预览帧提取等
与具体模型无关的部分收拢为 Mixin，供各视频节点复用。
"""

import base64
import io
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from .logging import logger

VIDEO_SYNTHESIS_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"

# 各地域的服务域名；带业务空间时使用业务空间专属域名（官方推荐，性能更好）
REGION_HOSTS = {
    "cn-beijing": "https://dashscope.aliyuncs.com",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com",
    "us": "https://dashscope-us.aliyuncs.com",
    "eu-central-1": "https://dashscope.aliyuncs.com",
}
WORKSPACE_HOST_REGIONS = ("cn-beijing", "ap-southeast-1", "eu-central-1")


class DashScopeVideoMixin:
    """DashScope 视频生成的公共方法，子类需定义 LOG_PREFIX"""

    LOG_PREFIX = "DashScope"

    # ----------------------------------------------------------------
    # 连接
    # ----------------------------------------------------------------

    def _resolve_api_key(self, api_key):
        """获取 API Key：节点参数优先，其次环境变量"""
        if api_key and api_key.strip():
            return api_key.strip()
        env_key = os.getenv("DASHSCOPE_API_KEY")
        if env_key:
            return env_key
        raise Exception("未设置 API Key，请在节点参数中输入或设置环境变量 DASHSCOPE_API_KEY")

    def _host(self, workspace_id, region):
        """根据地域与业务空间拼接服务域名"""
        if workspace_id and workspace_id.strip() and region in WORKSPACE_HOST_REGIONS:
            return f"https://{workspace_id.strip()}.{region}.maas.aliyuncs.com"
        return REGION_HOSTS.get(region, REGION_HOSTS["cn-beijing"])

    def _build_endpoint(self, workspace_id, region):
        """创建任务的请求端点"""
        return f"{self._host(workspace_id, region)}{VIDEO_SYNTHESIS_PATH}"

    def _build_task_url(self, task_id, workspace_id, region):
        """任务查询 URL"""
        return f"{self._host(workspace_id, region)}/api/v1/tasks/{task_id}"

    def _submit_task(self, request_body, api_key, workspace_id, region):
        """创建异步任务，返回 task_id"""
        endpoint = self._build_endpoint(workspace_id, region)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # 缺少此头会报 current user api does not support synchronous calls
            "X-DashScope-Async": "enable",
        }

        response = requests.post(endpoint, headers=headers, json=request_body, timeout=30)
        if response.status_code != 200:
            raise Exception(f"提交任务失败: HTTP {response.status_code} - {response.text}")

        result = response.json()
        if "code" in result:
            raise Exception(f"API 错误 [{result.get('code')}]: {result.get('message', '未知错误')}")

        task_id = result.get("output", {}).get("task_id")
        if not task_id:
            import json
            raise Exception(f"响应中未找到 task_id: {json.dumps(result, ensure_ascii=False)[:500]}")

        logger.info(f"[{self.LOG_PREFIX}] 任务已提交, task_id: {task_id}")
        return task_id

    def _poll_task_result(self, task_id, api_key, workspace_id, region, timeout, poll_interval):
        """轮询任务结果，返回最终响应（SUCCEEDED / FAILED）"""
        task_url = self._build_task_url(task_id, workspace_id, region)
        headers = {"Authorization": f"Bearer {api_key}"}
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"轮询超时: 已等待 {int(elapsed)}s，超过 {timeout}s 限制")

            try:
                logger.info(f"[{self.LOG_PREFIX}] 轮询任务状态: {task_id} ({int(elapsed)}s)")

                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()
                task_status = result.get("output", {}).get("task_status", "")

                logger.info(f"[{self.LOG_PREFIX}] 任务状态: {task_status}")

                if task_status in ("SUCCEEDED", "FAILED"):
                    return result
                if task_status == "CANCELED":
                    raise Exception("任务已被取消")
                if task_status == "UNKNOWN":
                    raise Exception("任务不存在或已过期 (UNKNOWN)，task_id 有效期为 24 小时")

                if task_status not in ("PENDING", "RUNNING"):
                    logger.warning(f"[{self.LOG_PREFIX}] 未知状态: {task_status}，继续轮询...")
                time.sleep(poll_interval)

            except requests.exceptions.RequestException as e:
                if time.time() - start_time > timeout:
                    raise Exception(f"轮询超时，最后网络错误: {str(e)}")
                logger.warning(f"[{self.LOG_PREFIX}] 网络错误: {str(e)}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

    # ----------------------------------------------------------------
    # 图像 / 媒体工具
    # ----------------------------------------------------------------

    def _image_tensor_to_base64(self, image):
        """将 ComfyUI IMAGE tensor 转换为 base64 (取第一张)"""
        if isinstance(image, torch.Tensor) and len(image.shape) == 4:
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

    def _image_to_url(self, image_tensor, image_url):
        """IMAGE 输入优先转 base64，否则取 URL；都没有则返回 None"""
        if image_tensor is not None:
            return f"data:image/png;base64,{self._image_tensor_to_base64(image_tensor)}"
        if image_url and image_url.strip():
            return image_url.strip()
        return None

    def _parse_urls(self, urls_text):
        """解析多行/逗号分隔的 URL 列表"""
        if not urls_text or not urls_text.strip():
            return []
        return [u.strip() for u in urls_text.replace(",", "\n").split("\n") if u.strip()]

    def _require_public_url(self, label, url):
        """音频/视频等不支持 base64 的素材，必须是公网 http(s) 或 oss 链接"""
        u = (url or "").strip()
        if not u:
            return ""
        if u.startswith("data:"):
            raise ValueError(f"{label} 不支持 base64 数据，请提供公网可访问的链接")
        if not (u.startswith("http://") or u.startswith("https://") or u.startswith("oss://")):
            raise ValueError(f"{label} 不合法: '{u[:100]}'，必须以 http:// 、https:// 或 oss:// 开头")
        return u

    # ----------------------------------------------------------------
    # 预览帧
    # ----------------------------------------------------------------

    def _extract_frame_from_video(self, video_url, timeout):
        """使用 ffmpeg 从视频 URL 提取第一帧作为预览"""
        # 提前检测 ffmpeg 是否存在，避免抛出 FileNotFoundError
        if shutil.which("ffmpeg") is None:
            logger.info(f"[{self.LOG_PREFIX}] 未检测到 ffmpeg，跳过预览帧提取（不影响视频生成）")
            return None
        try:
            temp_dir = Path(tempfile.gettempdir())
            timestamp = int(time.time())
            prefix = self.LOG_PREFIX.lower()
            video_file = temp_dir / f"{prefix}_video_{timestamp}.mp4"
            frame_file = temp_dir / f"{prefix}_frame_{timestamp}.png"

            try:
                logger.info(f"[{self.LOG_PREFIX}] 下载视频用于帧提取...")
                response = requests.get(video_url, timeout=timeout)
                if response.status_code != 200:
                    logger.warning(f"[{self.LOG_PREFIX}] 下载视频失败: HTTP {response.status_code}")
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
                video_file.unlink(missing_ok=True)
                frame_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"[{self.LOG_PREFIX}] 提取视频帧失败: {str(e)}")

        return None

    def _create_placeholder_frame(self):
        """创建占位预览图"""
        placeholder = np.full((720, 1280, 3), 0.2, dtype=np.float32)
        return torch.from_numpy(placeholder).unsqueeze(0)
