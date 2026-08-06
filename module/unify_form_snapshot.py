"""中台 /form 接口的模型清单快照（自动生成，请勿手改）

生成方式：调用 GET {base_url}/form 后按 modelSource/modelName 归档，
只保留 modelName / generateType / fields 三项，用作离线兜底基线。
运行期打开节点的 refresh_models 会拉取最新清单并覆盖本快照。

快照时间：FETCHED_AT 见下方常量。
"""

FETCHED_AT = "2026-08-06"

# [(modelSource, modelName, (支持的生成类型...), (表单字段...)), ...]
MODELS = [
    # 万相-图生视频2.6
    ('dashscope', 'wan2.6-i2v', ('I2V',), ('medias', 'prompt', 'negativePrompt', 'resolution', 'duration', 'promptExtend', 'watermark', 'generateAudio')),
    # 万相-文生视频2.6
    ('dashscope', 'wan2.6-t2v', ('T2V',), ('medias', 'prompt', 'negativePrompt', 'resolution', 'duration', 'promptExtend', 'watermark', 'generateAudio')),
    # 万相-参考生视频2.6
    ('dashscope', 'wan2.6-r2v', ('R2V',), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'promptExtend', 'watermark')),
    # 万相-图生视频2.7
    ('dashscope', 'wan2.7-i2v', ('I2V',), ('medias', 'prompt', 'negativePrompt', 'resolution', 'duration', 'promptExtend', 'watermark', 'generateAudio')),
    # 万相-文生视频2.7
    ('dashscope', 'wan2.7-t2v', ('T2V',), ('medias', 'prompt', 'negativePrompt', 'resolution', 'duration', 'promptExtend', 'watermark', 'generateAudio')),
    # 万相-参考生视频2.7
    ('dashscope', 'wan2.7-r2v', ('R2V',), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'promptExtend', 'watermark')),
    # 万相-视频编辑
    ('dashscope', 'wan2.7-videoedit', ('V2V',), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'promptExtend', 'audioSetting', 'watermark')),
    # 万相-视频3.0
    ('dashscope', 'wan3.0-video', ('I2V', 'T2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'duration', 'promptExtend', 'watermark', 'generateAudio')),
    # 快乐马-图生视频1.0
    ('dashscope', 'happyhorse-1.0-i2v', ('I2V',), ('medias', 'prompt', 'resolution', 'duration', 'watermark')),
    # 快乐马-文生视频1.0
    ('dashscope', 'happyhorse-1.0-t2v', ('T2V',), ('prompt', 'resolution', 'duration', 'watermark', 'ratio')),
    # 快乐马-参考生视频1.0
    ('dashscope', 'happyhorse-1.0-r2v', ('R2V',), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'watermark')),
    # 快乐马-视频编辑1.0
    ('dashscope', 'happyhorse-1.0-video-edit', ('V2V',), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'audioSetting', 'watermark')),
    # 快乐马-图生视频1.0
    ('dashscope', 'happyhorse-1.0-i2v-sg', ('I2V',), ('medias', 'prompt', 'resolution', 'duration', 'watermark')),
    # 快乐马-文生视频1.0
    ('dashscope', 'happyhorse-1.0-t2v-sg', ('T2V',), ('prompt', 'resolution', 'duration', 'watermark', 'ratio')),
    # 快乐马-参考生视频1.0
    ('dashscope', 'happyhorse-1.0-r2v-sg', ('R2V',), ('medias', 'resolution', 'ratio', 'duration', 'watermark')),
    # 快乐马-视频编辑1.0
    ('dashscope', 'happyhorse-1.0-video-edit-sg', ('V2V',), ('medias', 'resolution', 'ratio', 'duration', 'audioSetting', 'watermark')),
    # MiniMax-Hailuo-2.3
    ('minimax', 'MiniMax-Hailuo-2.3', ('I2V', 'T2V'), ('medias', 'prompt', 'negativePrompt', 'promptExtend', 'duration', 'resolution', 'watermark')),
    # MiniMax-H3
    ('minimax', 'MiniMax-H3', ('T2V', 'I2V', 'R2V'), ('medias', 'prompt', 'resolution', 'duration', 'ratio', 'watermark')),
    # kling-v2-5-turbo
    ('kling', 'kling-v2-5-turbo', ('I2V', 'T2V'), ('medias', 'prompt', 'negativePrompt', 'mode', 'cfgScale', 'multiShot', 'ratio', 'shotType', 'multiPrompts', 'generateAudio', 'duration', 'watermark', 'cameraType', 'horizontal', 'vertical', 'pan', 'tilt', 'roll', 'zoom')),
    # kling-v3
    ('kling', 'kling-v3', ('I2V', 'T2V'), ('medias', 'prompt', 'negativePrompt', 'mode', 'cfgScale', 'multiShot', 'ratio', 'shotType', 'multiPrompts', 'generateAudio', 'duration', 'watermark', 'cameraType', 'horizontal', 'vertical', 'pan', 'tilt', 'roll', 'zoom')),
    # kling-v3-omni
    ('kling', 'kling-v3-omni', ('R2V',), ('medias', 'prompt', 'negativePrompt', 'mode', 'cfgScale', 'multiShot', 'ratio', 'shotType', 'multiPrompts', 'generateAudio', 'duration', 'watermark', 'cameraType', 'horizontal', 'vertical', 'pan', 'tilt', 'roll', 'zoom')),
    # 可灵3.0-Turbo
    ('kling', 'kling-3.0-turbo', ('I2V', 'T2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'watermark')),
    # Seedance_2.0
    ('seedream', 'doubao-seedance-2-0-260128', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
    # Seedance_2.0_fast
    ('seedream', 'doubao-seedance-2-0-fast-260128', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
    # Seedance_2.0_mini
    ('seedream', 'doubao-seedance-2-0-mini-260615', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
    # Seedance_2.0
    ('idealab', 'DSD-2.0', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
    # Seedance_2.0_fast
    ('idealab', 'DSD-2.0-fast', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
    # Seedance_2.0_mini
    ('idealab', 'DSD-2.0-mini', ('T2V', 'I2V', 'R2V', 'V2V'), ('medias', 'prompt', 'resolution', 'ratio', 'duration', 'generateAudio', 'tools')),
]
