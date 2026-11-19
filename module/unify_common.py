"""中台统一接口的公共网络能力

只抽取视频与图片完全一致的基础设施：
- base_url 规范化及 /generate、/query 地址拼接
- token + custom_headers 请求头合并
- JSON 响应诊断
- 提交异步任务并提取 taskId

请求体、轮询响应字段、媒体处理和结果提取仍由各节点独立实现。
"""

import copy
import json

import requests

from .dashscope_video import merge_custom_headers


def resolve_unify_urls(base_url, default_base=""):
    """按中台统一约定生成提交和查询地址。

    用户填写完整的服务基础地址，例如：
    https://host/open/api/agent/v1/nhci/image/unify

    返回该地址追加 /generate 和 /query 后的两个端点。为兼容误填完整端点，
    会先移除末尾的 /generate 或 /query。
    """
    api_base = (base_url or "").strip().rstrip("/") or (default_base or "").strip().rstrip("/")
    if not api_base:
        raise ValueError("未填写 base_url，请填写中台 API 基础地址")

    for suffix in ("/generate", "/query"):
        if api_base.endswith(suffix):
            api_base = api_base[:-len(suffix)]
            break

    if not (api_base.startswith("http://") or api_base.startswith("https://") or api_base.startswith("/")):
        raise ValueError(f"base_url 不合法: '{api_base}'，需填写 http(s) 地址或站内绝对路径")

    return api_base, f"{api_base}/generate", f"{api_base}/query"


def build_unify_headers(token="", custom_headers="", log_prefix="Unify"):
    """构建中台请求头，自定义头最后合并并可覆盖内置值。"""
    headers = {"Content-Type": "application/json"}
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    return merge_custom_headers(headers, custom_headers, log_prefix)


def parse_json_response(response, action, base_url=""):
    """解析 JSON；遇到登录页、网关页或空响应时给出可诊断错误。"""
    try:
        return response.json()
    except ValueError:
        snippet = (response.text or "").strip()[:300] or "(空响应体)"
        hint = f" | base_url: {base_url}" if base_url else ""
        raise Exception(
            f"{action}返回的不是 JSON | HTTP {response.status_code} | "
            f"Content-Type: {response.headers.get('Content-Type')} | URL: {response.url}{hint} | "
            f"内容: {snippet}。请检查 base_url 以及 token / custom_headers"
        )


def add_supplier_credential(body, supplier_credential=""):
    """将可选供应商凭证 JSON 对象加入请求体，留空时不下发。"""
    if supplier_credential is None or supplier_credential == "":
        return body
    if isinstance(supplier_credential, str):
        if not supplier_credential.strip():
            return body
        try:
            credential = json.loads(supplier_credential)
        except ValueError:
            raise ValueError("supplier_credential 格式错误，请填写有效的 JSON 对象") from None
    else:
        credential = supplier_credential
    if not isinstance(credential, dict):
        raise ValueError("supplier_credential 必须是 JSON 对象")
    if credential:
        body["supplierCredential"] = credential
    return body


def merge_custom_body(body, custom_body=""):
    """递归合并自定义 JSON 对象；自定义值优先，数组和非对象值整体替换。"""
    if custom_body is None or custom_body == "":
        return body
    if isinstance(custom_body, str):
        if not custom_body.strip():
            return body
        try:
            extra = json.loads(custom_body)
        except ValueError:
            raise ValueError("custom_body 格式错误，请填写有效的 JSON 对象") from None
    else:
        extra = custom_body
    if not isinstance(extra, dict):
        raise ValueError("custom_body 必须是 JSON 对象")

    def merge(base, incoming):
        result = copy.deepcopy(base)
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    return merge(body, extra)


def redact_supplier_credential(body):
    """递归隐藏凭证字段，返回日志副本且不修改实际请求体。"""
    sensitive = {
        "suppliercredential", "token", "accesstoken", "refreshtoken", "apikey",
        "secret", "clientsecret", "accesskeysecret", "password", "authorization",
        "credential", "credentials", "signature", "privatekey",
    }

    def redact(value):
        if isinstance(value, dict):
            return {
                key: "[已隐藏]" if str(key).lower().replace("_", "").replace("-", "") in sensitive
                else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(body)


def submit_unify_task(generate_url, request_body, headers, timeout=30):
    """提交中台异步任务并返回 taskId。"""
    response = requests.post(generate_url, headers=headers, json=request_body, timeout=timeout)
    if response.status_code != 200:
        raise Exception(f"提交任务失败: HTTP {response.status_code} - {response.text}")

    result = parse_json_response(response, "提交任务", generate_url)
    if not result.get("success"):
        raise Exception(f"提交任务失败: [{result.get('code')}] {result.get('message')}")

    task_id = result.get("data")
    if not task_id:
        raise Exception(f"响应中未找到 taskId: {json.dumps(result, ensure_ascii=False)[:500]}")
    return task_id
