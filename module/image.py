"""中台统一图片生成节点

对接 docs/local/image.md（提交与视频一致，查询接口与视频不同）：
POST {prefix}/image/unify/generate  提交任务，data 为 taskId
GET  {prefix}/aigc/unify/getResult?taskId= 查询结果，data.images[].url 为生成图片
其中 base_url = {prefix}/image/unify，域名与 prefix 均取自用户填写的 base_url（如 /open/api/agent/v1/nhci）。

与视频节点不同，生图接口没有 generateType：有无参考图 images 即区分文生图 / 图生图，
因此节点只需选模型源与模型名。模型清单与视频的 model_registry 完全独立。
参考图只接受公网 URL（中台不支持 base64），与视频节点的 ref_image_urls 保持一致。
"""

import io
import json
import time

import numpy as np
import requests
import torch
from PIL import Image

from .logging import logger
from .unify_common import (
    add_supplier_credential,
    merge_custom_body,
    redact_supplier_credential,
    build_unify_headers,
    parse_json_response,
    resolve_unify_urls,
    submit_unify_task,
)

DEFAULT_HOST = ""

# 模型清单以对接文档 docs/local/image.md 的「支持的模型」表为准，modelSource 原样保留文档写法；
# 中台 /form 接口已停止更新，不要拿它的数据覆盖这里。
# 清单外的模型名可直接填在 model_name 里直通提交
IMAGE_MODELS = {
    "Seedream": ["doubao-seedream-4-5-251128", "jimeng_t2i_v40"],
    "Flux": ["flux-2-flex", "flux-2-pro-preview", "flux-2-max",
             "flux-2-klein-9b-preview", "flux-2-klein-4b"],
    "Wan": ["wan2.7-image", "wan2.7-image-pro"],
    "Qwen": ["qwen-image-2.0"],
    "Google": ["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite-image"],
    "azure": ["gpt-image-2"],
    "idealab": ["gpt-image-2"],
    "minimax": ["gemini-3.1-flash-image-preview", "gpt-image-2", "gpt-image-2.5-flare",
                "gpt-image-2.5-sunburst", "gemini-3.1-flash-lite-image"],
}


class UnifyImageGeneration:
    """中台统一图片生成节点

    - 无参考图 → 文生图；填了参考图 URL → 图生图 / 图片编辑
    - model_name 留空时取该模型源清单里的第一个；清单外的名字原样直通提交
    - 输出所有生成图组成的 IMAGE batch，以及换行分隔的图片 URL
    """

    @classmethod
    def INPUT_TYPES(cls):
        sources = list(IMAGE_MODELS.keys())
        default_source = "idealab"
        all_names = sorted({name for names in IMAGE_MODELS.values() for name in names})
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一只在草地上奔跑的金毛犬，阳光明媚，高清摄影",
                    "multiline": True
                }),
                "model_source": (sources, {"default": default_source}),
                "model_name": ("STRING", {
                    "default": IMAGE_MODELS[default_source][0],
                    "placeholder": "modelName，留空取该模型源的第一个；可选: " + ", ".join(all_names)
                }),
                "base_url": ("STRING", {
                    "default": DEFAULT_HOST,
                    "placeholder": "中台API基础地址，如 https://your-domain.com/api/image/unify"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "反向提示词"}),
                # auto 表示不下发，由模型自行决定
                "resolution": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"],
                                 {"default": "auto"}),
                # 0 表示不下发；宽高需同时填写
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "image_num": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                # --- 参考图 (图生图 / 编辑)，仅支持公网URL ---
                "ref_image_urls": ("STRING", {"default": "", "multiline": True, "placeholder": "参考图片URL，多个换行分隔"}),
                # --- 调用配置 ---
                "token": ("STRING", {"default": "", "placeholder": "Bearer Token"}),
                "source": ("STRING", {"default": "", "placeholder": "业务方来源标识，如 COMFY_AGENT"}),
                "operator": ("STRING", {"default": "", "placeholder": "操作者ID"}),
                "biz_id": ("STRING", {"default": "", "placeholder": "业务ID（幂等）"}),
                "callback_data": ("STRING", {"default": "", "placeholder": "回调数据，完成后原样返回"}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 3600, "step": 30}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "custom_headers": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": '自定义请求头，JSON 对象 {"X-Foo":"bar"} 或每行 Key: Value；与内置头合并，同名覆盖'
                }),
                # 留空时由 base_url 推导；新增项放末尾，避免旧工作流 widgets_values 错位
                "poll_url": ("STRING", {
                    "default": "",
                    "placeholder": "可选：完整轮询地址，留空按 base_url 推导为 .../aigc/unify/getResult"
                }),
                "supplier_credential": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": '可选供应商凭证 JSON，如 {"type":"BEARER","token":"..."}'
                }),
                "custom_body": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "可选请求体 JSON 对象；最后递归合并，同名覆盖，数组整体替换"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "info")
    FUNCTION = "generate"
    CATEGORY = "Malette/Image"

    def generate(self, prompt, model_source, model_name="", base_url=DEFAULT_HOST,
                 negative_prompt="", resolution="auto", aspect_ratio="auto",
                 width=0, height=0, image_num=1, ref_image_urls="",
                 token="", source="", operator="", biz_id="", callback_data="",
                 timeout=300, poll_interval=3, custom_headers="", poll_url="",
                 supplier_credential="", custom_body=""):
        try:
            generate_url, result_url = self._resolve_api_urls(base_url, poll_url)
            headers = build_unify_headers(token, custom_headers, "UnifyImage")

            model_name = self._resolve_model_name(model_source, model_name)

            request_body = self._build_request_body(
                prompt=prompt, model_source=model_source, model_name=model_name,
                negative_prompt=negative_prompt, resolution=resolution,
                aspect_ratio=aspect_ratio, width=width, height=height, image_num=image_num,
                ref_image_urls=ref_image_urls,
                source=source, operator=operator, biz_id=biz_id, callback_data=callback_data,
                supplier_credential=supplier_credential, custom_body=custom_body,
            )
            mode_desc = "图生图" if request_body.get("images") else "文生图"

            logger.info(f"[UnifyImage] 提交任务 | {model_source}/{model_name} | {mode_desc}")
            logger.info(f"[UnifyImage] 请求体: {json.dumps(redact_supplier_credential(request_body), ensure_ascii=False)}")

            # 步骤1: 提交任务
            logger.info(f"[UnifyImage] 提交地址: {generate_url}")
            task_id = submit_unify_task(generate_url, request_body, headers)

            logger.info(f"[UnifyImage] 任务已提交, taskId: {task_id}")

            # 步骤2: 轮询结果
            logger.info(f"[UnifyImage] 查询地址: {result_url}")
            data = self._poll_task_result(result_url, task_id, headers, timeout, poll_interval)

            status = data.get("taskStatus", "")
            if status != "DONE":
                raise Exception(f"任务失败 [{status}]: {data.get('errMsg') or '任务执行失败'}")

            urls = [item.get("url") for item in (data.get("images") or [])
                    if isinstance(item, dict) and item.get("url")]
            if not urls:
                raise Exception("任务成功但未返回图片 URL")

            logger.info(f"[UnifyImage] 生成成功，共 {len(urls)} 张")

            images = self._download_images(urls, timeout)
            h, w = images.shape[1], images.shape[2]
            info = (f"[{mode_desc}] 模型: {model_source}/{model_name} | "
                    f"张数: {len(urls)} | 尺寸: {w}x{h} | taskId: {task_id}")

            return (images, "\n".join(urls), info)

        except Exception as e:
            error_msg = f"图片生成失败: {str(e)}"
            logger.error(f"[UnifyImage] {error_msg}")
            raise Exception(error_msg)

    # ----------------------------------------------------------------
    # 参数解析
    # ----------------------------------------------------------------

    def _resolve_api_urls(self, base_url, poll_url=""):
        """生成提交与轮询地址

        poll_url 填写时原样使用；留空时保留 base_url 的域名与前缀，
        {prefix}/image/unify 推导为 {prefix}/aigc/unify/getResult。
        """
        api_base, generate_url, _ = resolve_unify_urls(base_url, DEFAULT_HOST)
        if not api_base.endswith("/image/unify"):
            raise ValueError(f"base_url 路径不合法: '{api_base}'，必须以 /image/unify 结尾")

        result_url = (poll_url or "").strip()
        if result_url:
            if not (result_url.startswith("http://") or result_url.startswith("https://") or result_url.startswith("/")):
                raise ValueError(f"poll_url 不合法: '{result_url}'，需填写完整 http(s) 地址或站内绝对路径")
            return generate_url, result_url

        prefix = api_base[:-len("/image/unify")]
        return generate_url, f"{prefix}/aigc/unify/getResult"

    def _resolve_model_name(self, model_source, model_name):
        """留空取清单第一个；清单外的名字直通提交并提示"""
        known = IMAGE_MODELS.get(model_source) or []
        name = (model_name or "").strip()
        if not name:
            if not known:
                raise ValueError(f"模型源 {model_source} 没有内置模型，请在 model_name 中填写")
            return known[0]
        if name not in known:
            logger.warning(f"[UnifyImage] {model_source} 清单中无 '{name}'（已知: {known}），按直通模式提交")
        return name

    # ----------------------------------------------------------------
    # 构建请求
    # ----------------------------------------------------------------

    def _build_request_body(self, prompt, model_source, model_name, negative_prompt,
                            resolution, aspect_ratio, width, height, image_num,
                            ref_image_urls,
                            source, operator, biz_id, callback_data, supplier_credential="",
                            custom_body=""):
        body = {
            "modelSource": model_source,
            "modelName": model_name,
            "prompt": prompt,
            "imageNum": image_num,
        }

        # 对接文档的 HTTP 字段表写的是 negPrompt（与 Java 类的 negativePrompt 不同），以 HTTP 文档为准
        if negative_prompt and negative_prompt.strip():
            body["negPrompt"] = negative_prompt.strip()
        if resolution and resolution != "auto":
            body["resolution"] = resolution
        if aspect_ratio and aspect_ratio != "auto":
            body["aspectRatio"] = aspect_ratio

        if width and height:
            body["width"] = width
            body["height"] = height
        elif width or height:
            logger.warning("[UnifyImage] width / height 需同时填写，已忽略单独设置的一项")

        images = self._parse_ref_urls(ref_image_urls)
        if images:
            body["images"] = images

        for key, value in (("source", source), ("operator", operator),
                           ("bizId", biz_id), ("callbackData", callback_data)):
            if value and value.strip():
                body[key] = value.strip()

        return merge_custom_body(add_supplier_credential(body, supplier_credential), custom_body)

    def _parse_ref_urls(self, ref_image_urls):
        """解析多行/逗号分隔的参考图 URL，提交前逐个校验（中台不支持 base64）"""
        urls = []
        for line in (ref_image_urls or "").replace(",", "\n").split("\n"):
            url = line.strip()
            if not url:
                continue
            if url.startswith("data:"):
                raise ValueError("参考图片URL 不支持 base64 数据，请提供公网可访问的 http(s) 链接")
            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError(f"参考图片URL 不合法: '{url[:100]}'，必须以 http:// 或 https:// 开头")
            urls.append(url)
        return urls

    # ----------------------------------------------------------------
    # 轮询
    # ----------------------------------------------------------------

    def _poll_task_result(self, url, task_id, headers, timeout, poll_interval):
        """轮询结果，状态处理与视频节点一致：DONE / ERROR 返回，处理中继续等"""
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"轮询超时: 已等待 {int(elapsed)}s，超过 {timeout}s 限制")

            try:
                response = requests.get(url, headers=headers, params={"taskId": task_id}, timeout=30)
                response.raise_for_status()
                # 非 JSON 响应不是网络抖动，直接报错而不是重试到超时
                result = parse_json_response(response, "查询结果", url)

                if not result.get("success"):
                    raise Exception(f"查询失败: [{result.get('code')}] {result.get('message')}")

                data = result.get("data") or {}
                status = data.get("taskStatus", "")

                logger.info(f"[UnifyImage] 任务状态: {status} ({int(elapsed)}s)")

                if status == "DONE":
                    return data
                elif status == "ERROR":
                    return data
                elif status == "PROCESSING" or status == "RUNNING":
                    time.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"[UnifyImage] 未知状态: {status}，继续轮询...")
                    time.sleep(poll_interval)
                    continue

            except requests.exceptions.RequestException as e:
                if time.time() - start_time > timeout:
                    raise Exception(f"轮询超时，最后网络错误: {str(e)}")
                logger.warning(f"[UnifyImage] 网络错误: {str(e)}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

    # ----------------------------------------------------------------
    # 图像工具
    # ----------------------------------------------------------------

    def _download_images(self, urls, timeout):
        """下载全部结果图拼成 batch；尺寸不一致时缩放到第一张的尺寸"""
        pil_images = []
        for url in urls:
            response = requests.get(url, timeout=min(timeout, 120))
            if response.status_code != 200:
                raise Exception(f"下载结果图失败: HTTP {response.status_code} - {url[:100]}")
            pil_images.append(Image.open(io.BytesIO(response.content)).convert("RGB"))

        size = pil_images[0].size
        tensors = []
        for img in pil_images:
            if img.size != size:
                logger.warning(f"[UnifyImage] 结果图尺寸不一致 {img.size}，缩放到 {size} 以组成 batch")
                img = img.resize(size, Image.LANCZOS)
            tensors.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0))
        return torch.stack(tensors)


NODE_CLASS_MAPPINGS = {
    "UnifyImageGeneration": UnifyImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifyImageGeneration": "中台统一图片生成 (Unify Image Generation)",
}
