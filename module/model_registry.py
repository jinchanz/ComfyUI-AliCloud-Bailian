"""视频模型注册表

**重要：两个调用通道彼此独立，绝不共享模型名。**

- channel="unify"     走中台统一接口 /api/video/unify。
                      模型源键必须与中台 modelSource 枚举一致，实测为：
                      dashscope / minimax / kling / seedream / idealab。
                      万相、快乐马(HappyHorse) 等模型在中台侧统一归到 dashscope 之下，
                      中台不接受 modelSource=wan / happyhorse。
- channel="dashscope" 直连阿里云百炼 DashScope 接口，没有 modelSource 概念，
                      只有百炼官方模型名（如 wan2.7-t2v、happyhorse-1.1-i2v）。
                      注意：百炼直连有 happyhorse-1.1，而中台侧目前只有 1.0。

unify 通道的基线数据来自 unify_form_snapshot（GET /form 的真实快照），
包含每个模型支持的生成类型与表单字段清单。

**注意：/form 的 fields[].id 是前端表单字段 id，并不等于请求体字段名。**
实测反例：表单里是 medias / watermark，请求体却必需是 mediaList / waterMark。
因此字段清单只能用作「模型能力的参考」与「补充声明」，不能拿它去删减请求体字段，
否则会把模型其实支持的参数（如万相 T2V 的 ratio）吞掉。

清单解析优先级：
1. unify_form_snapshot 快照 + 内置友好版本别名 —— 兜底，离线可用；
2. 中台 /form 运行期同步（refresh_models）—— 只影响 unify 通道；
3. 本地覆盖文件 video_models.json —— 顶层必须是通道名，按 mtime 热加载；
4. version 直通 —— 清单里没有的版本原样当作模型名提交。
"""

import json
import os
import time
from pathlib import Path

import requests

from .logging import logger
from .unify_form_snapshot import MODELS as SNAPSHOT_MODELS

# 通道
CHANNEL_UNIFY = "unify"
CHANNEL_DASHSCOPE = "dashscope"

# 规范化后的生成模式
MODE_T2V = "T2V"
MODE_I2V = "I2V"
MODE_R2V = "R2V"
MODE_VIDEOEDIT = "VIDEOEDIT"
ALL_MODES = (MODE_T2V, MODE_I2V, MODE_R2V, MODE_VIDEOEDIT)

# 各种写法 → 规范模式（中台用 V2V，HappyHorse 节点内部用 video_edit）
MODE_ALIASES = {
    "t2v": MODE_T2V,
    "text2video": MODE_T2V,
    "i2v": MODE_I2V,
    "image2video": MODE_I2V,
    "r2v": MODE_R2V,
    "reference2video": MODE_R2V,
    "v2v": MODE_VIDEOEDIT,
    "videoedit": MODE_VIDEOEDIT,
    "video_edit": MODE_VIDEOEDIT,
    "video-edit": MODE_VIDEOEDIT,
}

# 版本表里的保留键（不会被当成生成模式）
RESERVED_VERSION_KEYS = ("extras", "fields", "*")

# 供应商专属参数：这些字段只在模型的表单字段里出现时才下发
VENDOR_EXTRA_FIELDS = ("mode", "cfgScale", "audioSetting")

# 中台各模型源的友好版本别名：把一组真实模型名按生成模式聚合成一个"版本"，
# 用户选版本即可，节点按检测到的模式挑出正确的模型名。
# 所有模型名均来自 /form 实测，未出现在快照中的名字会被下面的自检测试拦住。
UNIFY_VERSION_ALIASES = {
    "dashscope": {
        "default_version": "wan-2.7",
        "versions": {
            "wan-2.6": {
                "T2V": "wan2.6-t2v",
                "I2V": "wan2.6-i2v",
                "R2V": "wan2.6-r2v",
            },
            "wan-2.7": {
                "T2V": "wan2.7-t2v",
                "I2V": "wan2.7-i2v",
                "R2V": "wan2.7-r2v",
                "VIDEOEDIT": "wan2.7-videoedit",
            },
            "wan-3.0": "wan3.0-video",
            "happyhorse-1.0": {
                "T2V": "happyhorse-1.0-t2v",
                "I2V": "happyhorse-1.0-i2v",
                "R2V": "happyhorse-1.0-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
            # 新加坡区域模型
            "happyhorse-1.0-sg": {
                "T2V": "happyhorse-1.0-t2v-sg",
                "I2V": "happyhorse-1.0-i2v-sg",
                "R2V": "happyhorse-1.0-r2v-sg",
                "VIDEOEDIT": "happyhorse-1.0-video-edit-sg",
            },
            # 兼容别名：中台侧目前没有 happyhorse-1.1，指向 1.0 的模型
            "happyhorse-1.1": {
                "T2V": "happyhorse-1.0-t2v",
                "I2V": "happyhorse-1.0-i2v",
                "R2V": "happyhorse-1.0-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
        },
    },
    "minimax": {
        "default_version": "MiniMax-Hailuo-2.3",
        "versions": {
            # 旧节点下拉里的版本值
            "2.3": "MiniMax-Hailuo-2.3",
        },
    },
    "kling": {
        "default_version": "kling-v3",
        # 可灵系列的专属参数，与旧版节点行为一致
        "extras": ["mode", "cfgScale"],
        "versions": {},
    },
    "seedream": {
        "default_version": "2.0",
        "versions": {
            # 旧节点下拉里的版本值
            "2.0": "doubao-seedance-2-0-260128",
            "fast": "doubao-seedance-2-0-fast-260128",
            "mini": "doubao-seedance-2-0-mini-260615",
        },
    },
    "idealab": {
        "default_version": "DSD-2.0",
    },
}

# 向后兼容的遗留模型源别名：旧版节点下拉里有 wan / happyhorse，
# 中台实际不接受这两个 modelSource；直接从下拉删掉会让存了旧值的工作流
# 卡在 ComfyUI 的 "value not in list" 校验上连跑都跑不了，故保留为别名，
# 提交时通过 request_source 换成 dashscope。
UNIFY_LEGACY_SOURCES = {
    "wan": {
        "legacy": True,
        "request_source": "dashscope",
        "default_version": "2.7",
        # 与旧版节点行为一致：遗留 wan 源一律下发 audioSetting
        "extras": ["audioSetting"],
        "versions": {
            "2.6": {
                "T2V": "wan2.6-t2v",
                "I2V": "wan2.6-i2v",
                "R2V": "wan2.6-r2v",
            },
            "2.7": {
                "T2V": "wan2.7-t2v",
                "I2V": "wan2.7-i2v",
                "R2V": "wan2.7-r2v",
                "VIDEOEDIT": "wan2.7-videoedit",
            },
            "3.0": "wan3.0-video",
        },
    },
    "happyhorse": {
        "legacy": True,
        "request_source": "dashscope",
        "default_version": "1.0",
        "versions": {
            # 中台侧只有 1.0，1.1 作为兼容别名同样指向 1.0
            "1.1": {
                "T2V": "happyhorse-1.0-t2v",
                "I2V": "happyhorse-1.0-i2v",
                "R2V": "happyhorse-1.0-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
            "1.0": {
                "T2V": "happyhorse-1.0-t2v",
                "I2V": "happyhorse-1.0-i2v",
                "R2V": "happyhorse-1.0-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
        },
    },
}

# 直连阿里云百炼：模型名以百炼官方 API 文档为准，与中台无关
DASHSCOPE_MODELS = {
    # HappyHorse：视频编辑能力仅 1.0 提供，故 1.1 的 VIDEOEDIT 指向 1.0 的模型
    "happyhorse": {
        "default_version": "1.1",
        "versions": {
            "1.1": {
                "T2V": "happyhorse-1.1-t2v",
                "I2V": "happyhorse-1.1-i2v",
                "R2V": "happyhorse-1.1-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
            "1.0": {
                "T2V": "happyhorse-1.0-t2v",
                "I2V": "happyhorse-1.0-i2v",
                "R2V": "happyhorse-1.0-r2v",
                "VIDEOEDIT": "happyhorse-1.0-video-edit",
            },
        },
    },
    # 万相：T2V 与 I2V 是两个不同模型，首尾帧与视频续写都由 i2v 模型承担
    # （靠 input.media 的素材组合区分），不支持 R2V
    "wan": {
        "default_version": "2.7",
        "versions": {
            "2.7": {
                "T2V": "wan2.7-t2v",
                "I2V": "wan2.7-i2v",
                "VIDEOEDIT": "wan2.7-i2v",
            },
            "2.7-snapshot": {
                "T2V": "wan2.7-t2v-2026-06-12",
                "I2V": "wan2.7-i2v-2026-04-25",
                "VIDEOEDIT": "wan2.7-i2v-2026-04-25",
            },
        },
    },
}

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
OVERRIDE_FILENAME = "video_models.json"
REMOTE_CACHE_PATH = PLUGIN_ROOT / ".cache" / "video_models_remote.json"
REMOTE_CACHE_TTL = 6 * 3600  # 远端清单缓存 6 小时


def normalize_mode(mode):
    """把各种模式写法规范成 T2V / I2V / R2V / VIDEOEDIT"""
    if not mode:
        return ""
    key = str(mode).strip().lower().replace(" ", "")
    return MODE_ALIASES.get(key, str(mode).strip().upper())


def _normalize_version_entry(value):
    """
    规范化单个版本项，返回 {"models": {模式: 模型名}, "extras": [...], "fields": [...]|None}。
    支持写法：
    - 字符串："wan3.0-video"                        → 所有模式共用该模型名
    - 模式字典：{"T2V": "x", "V2V": "y"}            → 按模式指定，"*" 兜底
    - 附加声明：{"*": "x", "extras": [...], "fields": [...]}
    """
    if isinstance(value, str):
        return {"models": {m: value for m in ALL_MODES}, "extras": [], "fields": None}
    if not isinstance(value, dict):
        return None

    fields = value.get("fields")
    models = {}
    for mode, model_name in value.items():
        if mode in RESERVED_VERSION_KEYS:
            continue
        canonical = normalize_mode(mode)
        if canonical:
            models[canonical] = model_name
    fallback = value.get("*")
    if fallback:
        for m in ALL_MODES:
            models.setdefault(m, fallback)
    return {
        "models": models,
        "extras": list(value.get("extras") or []),
        "fields": list(fields) if fields is not None else None,
    }


def _normalize_source(entry):
    """兼容两种写法：{"版本": ...} 扁平写法，或 {"versions": {...}, ...}"""
    if not isinstance(entry, dict):
        return None
    detailed = "versions" in entry
    raw_versions = entry.get("versions") if detailed else entry

    versions = {}
    for version, value in (raw_versions or {}).items():
        normalized = _normalize_version_entry(value)
        if normalized is not None:
            versions[str(version)] = normalized

    return {
        "versions": versions,
        "extras": list(entry.get("extras") or []) if detailed else [],
        "default_version": (entry.get("default_version") or "") if detailed else "",
        # 遗留别名：提交时实际下发的 modelSource（缺省就是模型源自己）
        "request_source": (entry.get("request_source") or "") if detailed else "",
        "legacy": bool(entry.get("legacy")) if detailed else False,
    }


def _merge_source(base, incoming):
    """合并同一模型源：版本表逐项覆盖，extras 取并集"""
    merged = {
        "versions": dict(base.get("versions") or {}),
        "extras": list(base.get("extras") or []),
        "default_version": base.get("default_version") or "",
        "request_source": base.get("request_source") or "",
        "legacy": bool(base.get("legacy")),
    }
    merged["versions"].update(incoming.get("versions") or {})
    for extra in incoming.get("extras") or []:
        if extra not in merged["extras"]:
            merged["extras"].append(extra)
    if incoming.get("default_version"):
        merged["default_version"] = incoming["default_version"]
    if incoming.get("request_source"):
        merged["request_source"] = incoming["request_source"]
    return merged


def _normalize_channels(raw):
    """规范化 {通道: {模型源: 配置}} 结构"""
    data = {}
    for channel, sources in (raw or {}).items():
        if not isinstance(sources, dict):
            continue
        bucket = {}
        for source, entry in sources.items():
            normalized = _normalize_source(entry)
            if normalized:
                bucket[str(source)] = normalized
        if bucket:
            data[str(channel)] = bucket
    return data


def _forms_to_layer(forms):
    """
    把 /form 风格的模型清单（modelSource / modelName / generateType / fields）
    转成 {模型源: {versions: {模型名: {...}}}} 以及 {模型名: 字段清单}。
    """
    sources = {}
    model_fields = {}
    for source, model_name, generate_types, fields in forms:
        if not source or not model_name:
            continue
        modes = [normalize_mode(m) for m in (generate_types or ALL_MODES)]
        entry = sources.setdefault(source, {"versions": {}})
        version_entry = {m: model_name for m in modes if m}
        if fields:
            version_entry["fields"] = list(fields)
            model_fields[model_name] = list(fields)
        entry["versions"][model_name] = version_entry
    return sources, model_fields


class VideoModelRegistry:
    """模型清单的加载、合并与查询（按通道隔离）"""

    def __init__(self):
        self._cache = None
        self._fields_cache = None
        self._override_mtime = None
        self._remote_loaded_at = 0

    # ------------------------------------------------------------------
    # 加载各层清单
    # ------------------------------------------------------------------

    def _override_path(self):
        env_path = os.getenv("COMFY_BAILIAN_VIDEO_MODELS")
        if env_path and env_path.strip():
            return Path(env_path.strip())
        return PLUGIN_ROOT / OVERRIDE_FILENAME

    def _load_override(self):
        """读取本地覆盖文件，返回 (清单, mtime)"""
        path = self._override_path()
        if not path.is_file():
            return {}, None
        try:
            mtime = path.stat().st_mtime
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 只接受 {通道: {模型源: ...}} 结构，避免误把模型源当通道
            raw = {k: v for k, v in (raw or {}).items() if k in (CHANNEL_UNIFY, CHANNEL_DASHSCOPE)}
            data = _normalize_channels(raw)
            total = sum(len(v) for v in data.values())
            logger.info(f"[ModelRegistry] 已加载本地模型清单: {path}（{total} 个模型源）")
            return data, mtime
        except Exception as e:
            logger.warning(f"[ModelRegistry] 本地模型清单解析失败，忽略: {path} - {e}")
            return {}, None

    def _load_remote_cache(self):
        """读取 /form 的本地缓存（只影响 unify 通道），返回 (清单, 字段表)"""
        if not REMOTE_CACHE_PATH.is_file():
            return {}, {}
        try:
            with open(REMOTE_CACHE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._remote_loaded_at = payload.get("fetched_at", 0)
            forms = [tuple(item) for item in (payload.get("forms") or [])]
            sources, model_fields = _forms_to_layer(forms)
            return _normalize_channels({CHANNEL_UNIFY: sources}), {CHANNEL_UNIFY: model_fields}
        except Exception as e:
            logger.warning(f"[ModelRegistry] 远端模型缓存读取失败，忽略: {e}")
            return {}, {}

    def refresh_from_remote(self, base_url, headers=None, timeout=15, force=False):
        """
        从中台 GET {base_url}/form 拉取当前支持的模型清单并落盘缓存。
        /form 是 modelName 级别的清单，因此以 modelName 作为版本键注册到 unify 通道，
        同时记录每个模型的表单字段用于请求体过滤。直连百炼通道不受影响。
        """
        if not base_url or not base_url.strip():
            return False

        if not force and (time.time() - self._remote_loaded_at) < REMOTE_CACHE_TTL:
            return False

        url = f"{base_url.strip().rstrip('/')}/form"
        try:
            response = requests.get(url, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.warning(f"[ModelRegistry] 拉取中台模型清单失败（回退到本地快照）: {url} - {e}")
            return False

        if not result.get("success"):
            logger.warning(f"[ModelRegistry] /form 返回失败: [{result.get('code')}] {result.get('message')}")
            return False

        forms = []
        for form in result.get("data") or []:
            source = form.get("modelSource")
            model_name = form.get("modelName")
            if not source or not model_name:
                continue
            fields = [f.get("id") for f in (form.get("fields") or []) if isinstance(f, dict) and f.get("id")]
            forms.append([source, model_name, list(form.get("generateType") or []), fields])

        if not forms:
            logger.warning("[ModelRegistry] /form 未返回任何模型，保留现有清单")
            return False

        try:
            REMOTE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REMOTE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"fetched_at": time.time(), "forms": forms}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[ModelRegistry] 远端模型清单写入缓存失败: {e}")

        self._remote_loaded_at = time.time()
        self._cache = None  # 使下次查询重新合并
        self._fields_cache = None
        sources = {f[0] for f in forms}
        logger.info(f"[ModelRegistry] 已从中台同步 {len(sources)} 个模型源 / {len(forms)} 个模型")
        return True

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------

    def _builtin(self):
        """快照 + 友好版本别名 + 遗留别名 + 直连通道，构成内置基线"""
        snapshot_sources, snapshot_fields = _forms_to_layer(SNAPSHOT_MODELS)
        unify = _normalize_channels({CHANNEL_UNIFY: snapshot_sources}).get(CHANNEL_UNIFY, {})

        for source, entry in UNIFY_VERSION_ALIASES.items():
            normalized = _normalize_source(entry if "versions" in entry else {"versions": {}, **entry})
            if source in unify:
                unify[source] = _merge_source(unify[source], normalized)
            else:
                unify[source] = normalized

        for source, entry in UNIFY_LEGACY_SOURCES.items():
            unify[source] = _normalize_source(entry)

        dashscope = _normalize_channels({CHANNEL_DASHSCOPE: DASHSCOPE_MODELS}).get(CHANNEL_DASHSCOPE, {})
        return {CHANNEL_UNIFY: unify, CHANNEL_DASHSCOPE: dashscope}, {CHANNEL_UNIFY: snapshot_fields}

    def _ensure_cache(self):
        """按 内置 < 远端缓存 < 本地覆盖 的优先级合并出最终清单（带热加载）"""
        path = self._override_path()
        current_mtime = path.stat().st_mtime if path.is_file() else None
        if self._cache is not None and current_mtime == self._override_mtime:
            return

        merged, fields = self._builtin()
        remote, remote_fields = self._load_remote_cache()
        override, _ = self._load_override()

        for layer in (remote, override):
            for channel, sources in layer.items():
                bucket = merged.setdefault(channel, {})
                for source, entry in sources.items():
                    if source in bucket:
                        bucket[source] = _merge_source(bucket[source], entry)
                    else:
                        bucket[source] = entry

        for channel, model_fields in remote_fields.items():
            fields.setdefault(channel, {}).update(model_fields)

        # 覆盖文件里显式声明的 fields 也纳入字段表
        for channel, sources in override.items():
            for entry in sources.values():
                for version_entry in (entry.get("versions") or {}).values():
                    declared = version_entry.get("fields")
                    if declared:
                        for model_name in (version_entry.get("models") or {}).values():
                            fields.setdefault(channel, {})[model_name] = list(declared)

        self._cache = merged
        self._fields_cache = fields
        self._override_mtime = current_mtime

    def _registry(self):
        self._ensure_cache()
        return self._cache

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _source_entry(self, channel, source):
        bucket = self._registry().get(channel) or {}
        entry = bucket.get(source)
        if not entry:
            raise ValueError(
                f"{channel} 通道不支持模型源 '{source}'，可选: {list(bucket.keys())}"
            )
        return entry

    def sources(self, channel):
        """
        某个通道的可选模型源。现行值排在前面、遗留别名排在最后；
        遗留别名必须保留在列表里，否则存了旧值的工作流会过不了 ComfyUI 的下拉校验。
        """
        bucket = self._registry().get(channel) or {}
        snapshot_order = [s for s, _, _, _ in SNAPSHOT_MODELS] if channel == CHANNEL_UNIFY else []
        preferred = []
        for s in snapshot_order + list(UNIFY_VERSION_ALIASES) + list(DASHSCOPE_MODELS):
            if s in bucket and s not in preferred:
                preferred.append(s)
        ordered = preferred + [s for s in bucket if s not in preferred]
        current = [s for s in ordered if not bucket[s].get("legacy")]
        legacy = [s for s in ordered if bucket[s].get("legacy")]
        return current + legacy

    def request_source(self, channel, source):
        """提交时实际应下发的模型源（遗留别名会被换成真实枚举值）"""
        bucket = self._registry().get(channel) or {}
        entry = bucket.get(source) or {}
        return entry.get("request_source") or source

    def versions(self, channel, source):
        """某个模型源的可选版本"""
        bucket = self._registry().get(channel) or {}
        entry = bucket.get(source) or {}
        return list((entry.get("versions") or {}).keys())

    def default_version(self, channel, source):
        bucket = self._registry().get(channel) or {}
        entry = bucket.get(source) or {}
        if entry.get("default_version"):
            return entry["default_version"]
        versions = self.versions(channel, source)
        return versions[0] if versions else ""

    def model_fields(self, channel, model_name):
        """
        某个模型的**表单**字段清单（来自 /form）。未知时返回 None。
        注意：这些 id 是前端表单字段名，与请求体字段名并非一一对应
        （medias→mediaList、watermark→waterMark），仅供能力参考。
        """
        self._ensure_cache()
        return (self._fields_cache.get(channel) or {}).get(model_name)

    def supports_field(self, channel, model_name, field):
        """该模型的表单是否列出了某个字段；清单未知时一律返回 True。
        仅用于诊断提示，**不要**拿它做请求体字段的取舍。"""
        fields = self.model_fields(channel, model_name)
        if fields is None:
            return True
        return field in fields

    def extras(self, channel, source, version="", model_name=""):
        """
        该模型需要下发的供应商专属参数（mode / cfgScale / audioSetting）。

        取「内置声明（模型源级 + 版本级）」与「/form 字段清单」的**并集**：
        内置声明保证既有行为不变，/form 只负责补充声明里漏掉的（如快乐马视频编辑的
        audioSetting）。不做删减，因为表单字段清单并非接口契约，删减会造成行为倒退。
        """
        bucket = self._registry().get(channel) or {}
        entry = bucket.get(source) or {}
        result = list(entry.get("extras") or [])

        version = (version or "").strip() or self.default_version(channel, source)
        version_entry = (entry.get("versions") or {}).get(version) or {}
        for extra in version_entry.get("extras") or []:
            if extra not in result:
                result.append(extra)

        if model_name:
            fields = self.model_fields(channel, model_name) or []
            for extra in VENDOR_EXTRA_FIELDS:
                if extra in fields and extra not in result:
                    result.append(extra)

        return result

    def resolve(self, channel, source, version, mode):
        """
        解析模型名。命中清单则返回映射值；
        清单里没有该版本时，把 version 原样当作模型名直通提交（并给出提示）。
        """
        canonical_mode = normalize_mode(mode)
        entry = self._source_entry(channel, source)

        version = (version or "").strip() or self.default_version(channel, source)
        if not version:
            raise ValueError(
                f"{channel}/{source} 清单为空且未填版本，请在版本处直接填写模型名，"
                f"或在 {OVERRIDE_FILENAME} 中补充清单"
            )

        version_entry = (entry.get("versions") or {}).get(version)
        if version_entry is None:
            logger.warning(
                f"[ModelRegistry] {channel}/{source} 清单中无版本 '{version}'"
                f"（已知: {self.versions(channel, source)}），按直通模式将其作为模型名提交"
            )
            return version

        models = version_entry.get("models") or {}
        model_name = models.get(canonical_mode)
        if model_name:
            return model_name

        # 该版本没声明这个模式：若整个版本只对应一个模型名，就沿用它并告警
        # （模型自身可能支持而 /form 未声明，硬报错会让原本能跑的工作流跑不了）；
        # 若不同模式对应不同模型名，则无法推断，直接报错。
        distinct = {name for name in models.values() if name}
        supported = [m for m in ALL_MODES if models.get(m)]
        if len(distinct) == 1:
            only = distinct.pop()
            logger.warning(
                f"[ModelRegistry] {channel}/{source} 的 {version} 未声明支持 {canonical_mode}"
                f"（已声明: {supported}），仍按 {only} 提交，若上游拒绝请改用其它版本"
            )
            return only

        raise ValueError(
            f"{channel}/{source} 的 {version} 不支持 {canonical_mode} 模式，仅支持: {supported}"
        )


# 全局单例：所有节点共用一份清单与缓存
registry = VideoModelRegistry()
