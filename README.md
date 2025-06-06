# ComfyUI 阿里云百炼 API 节点

这是一个用于在 ComfyUI 中调用阿里云阿里云百炼（DashScope）API 的自定义节点集合。

## 功能特性

- 支持调用阿里云百炼图像合成 API
- 支持异步/同步模式
- 完整的错误处理和日志记录
- 可配置的 API 端点和参数
- 提供三种不同粒度的节点：完整流程、任务提交、结果轮询

## 安装

1. 将插件文件夹放置在 ComfyUI 的 `custom_nodes` 目录下
2. 重启 ComfyUI
3. 确保已安装 `requests` 库：`pip install requests`

## 节点说明

### 1. 阿里云百炼 API 调用 (BailianAPI)
**完整的一体化节点，自动处理整个流程**

#### 输入参数
**必需参数：**
- `endpoint`: API 端点 URL（默认为阿里云百炼图像合成 API）
- `params`: JSON 格式的请求参数

**可选参数：**
- `api_key`: 您的阿里云 API 密钥
- `model`: 使用的模型名称（默认为 "aitryon-plus"）
- `async_mode`: 是否启用异步模式（默认为 true）
- `poll_interval`: 轮询间隔时间（秒，默认为 3 秒）
- `max_wait_time`: 最大等待时间（秒，默认为 300 秒）

#### 输出
- `response`: 完整的 API 响应结果

### 2. 阿里云百炼 API 提交任务 (BailianAPISubmit)
**只负责提交任务，返回 task_id 用于后续轮询**

#### 输入参数
**必需参数：**
- `endpoint`: API 端点 URL
- `params`: JSON 格式的请求参数

**可选参数：**
- `api_key`: 您的阿里云 API 密钥
- `model`: 使用的模型名称（默认为 "aitryon-plus"）
- `async_mode`: 是否启用异步模式（默认为 true）

#### 输出
- `task_id`: 任务 ID（用于轮询）
- `response`: 提交请求的响应

### 3. 阿里云百炼 API 轮询结果 (BailianAPIPoll)
**根据 task_id 轮询获取任务结果**

#### 输入参数
**必需参数：**
- `task_id`: 要查询的任务 ID

**可选参数：**
- `api_key`: 您的阿里云 API 密钥
- `poll_interval`: 轮询间隔时间（秒，默认为 3 秒）
- `max_wait_time`: 最大等待时间（秒，默认为 300 秒）
- `single_query`: 是否只查询一次（默认为 false，会持续轮询直到完成）

#### 输出
- `result`: 任务结果或状态信息
- `status`: 任务状态（SUCCEEDED/FAILED/PENDING/RUNNING/TIMEOUT/ERROR）

### 4. JSON 键值提取器 (JSONExtractor)
**从JSON数据中提取嵌套的键值**

#### 输入参数
**必需参数：**
- `json_input`: 输入的JSON字符串
- `key_path`: 要提取的键路径（支持嵌套，如 "output.image_url"）

**可选参数：**
- `default_value`: 当键不存在时返回的默认值（默认为空字符串）
- `return_as_string`: 是否将结果转换为字符串（默认为 true）

#### 输出
- `value`: 提取到的值

#### 支持的路径格式
- 简单键：`"image_url"`
- 嵌套键：`"output.image_url"`
- 数组索引：`"items.0.name"`（获取数组第一个元素的name字段）
- 复杂嵌套：`"data.results.0.metadata.url"`

### 5. JSON 键值修改器 (JSONModifier)
**修改JSON数据中的嵌套键值**

#### 输入参数
**必需参数：**
- `json_input`: 输入的JSON字符串
- `key_path`: 要修改的键路径（支持嵌套，如 "parameters.restore_face"）
- `new_value`: 新的值

**可选参数：**
- `value_type`: 值类型（auto/string/number/boolean/json，默认为auto自动检测）
- `create_path`: 当路径不存在时是否创建（默认为true）

#### 输出
- `modified_json`: 修改后的JSON字符串

#### 支持的值类型
- **auto**：自动检测类型（推荐）
- **string**：强制转换为字符串
- **number**：转换为数字（整数或浮点数）
- **boolean**：转换为布尔值（true/false、1/0、yes/no等）
- **json**：解析为JSON对象或数组

### 参数格式示例

#### API请求参数示例
```json
{
  "input": {
    "top_garment_url": "http://example.com/top.jpg",
    "bottom_garment_url": "http://example.com/bottom.jpg",
    "person_image_url": "http://example.com/person.jpg"
  },
  "parameters": {
    "resolution": -1,
    "restore_face": true
  }
}
```

#### JSON提取器使用示例
对于您提供的API响应：
```json
{
  "request_id": "57aca18e-b8d1-96bf-ad74-bef5625ede1f",
  "output": {
    "task_id": "cb707f41-3c43-46e6-a8fc-47055b299c9b",
    "task_status": "SUCCEEDED",
    "image_url": "http://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1d/51/20250606/3080f59a/cb707f41-3c43-46e6-a8fc-47055b299c9b_tryon.jpg?Expires=1749294210&OSSAccessKeyId=LTAI5tKPD3TMqf2Lna1fASuh&Signature=1jc8rhExY4KZ6Pv%2B1yLwDczod0E%3D"
  },
  "usage": {
    "image_count": 1
  }
}
```

使用JSON键值提取器：
- 提取图片URL：`key_path = "output.image_url"`
- 提取任务状态：`key_path = "output.task_status"`
- 提取请求ID：`key_path = "request_id"`
- 提取使用统计：`key_path = "usage.image_count"`

#### JSON修改器使用示例
对于您提供的API请求JSON：
```json
{
    "model": "aitryon-plus",
    "input": {
        "top_garment_url": "https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/2389646171/p801332.jpeg",
        "bottom_garment_url": "https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1389646171/p801326.jpeg",
        "person_image_url": "https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1389646171/p801328.png"
    },
    "parameters": {
        "resolution": -1,
        "restore_face": true
    }
}
```

使用JSON键值修改器：
- 修改restore_face：`key_path = "parameters.restore_face"`, `new_value = "false"`
- 修改上衣图片：`key_path = "input.top_garment_url"`, `new_value = "新的图片URL"`
- 修改分辨率：`key_path = "parameters.resolution"`, `new_value = "1024"`
- 修改模型：`key_path = "model"`, `new_value = "新模型名称"`

## 使用场景

### 场景一：简单使用（推荐新手）
使用 **阿里云百炼 API 调用** 节点，一步到位完成整个流程。

### 场景二：分步骤控制（推荐高级用户）
1. 使用 **阿里云百炼 API 提交任务** 节点提交任务
2. 将 `task_id` 输出连接到 **阿里云百炼 API 轮询结果** 节点
3. 可以在中间插入其他逻辑，如延时、条件判断等

### 场景三：状态检查
使用 **阿里云百炼 API 轮询结果** 节点的 `single_query` 模式，只查询一次任务状态而不持续轮询。

### 场景四：提取特定数据
将API响应连接到 **JSON 键值提取器** 节点，快速提取需要的字段，如图片URL、任务状态等。

### 场景五：动态修改参数
使用 **JSON 键值修改器** 节点动态修改API请求参数，如更换图片URL、调整参数等，无需手动编辑整个JSON。

## 异步任务处理

当启用异步模式时：

1. 首次请求获取 `task_id` 和初始状态
2. 如果状态为 `PENDING`，开始轮询任务状态
3. 每隔 `poll_interval` 秒查询一次任务状态
4. 直到任务状态变为 `SUCCEEDED`（成功）或 `FAILED`（失败）
5. 超过 `max_wait_time` 时间后自动超时

## 注意事项

1. 需要有效的阿里云 API 密钥才能正常使用
2. 图像 URL 需要是公开可访问的
3. 异步模式下会自动轮询，无需手动处理任务状态
4. 建议根据任务复杂度调整轮询间隔和最大等待时间

## 错误处理

节点包含完整的错误处理机制：
- 网络请求错误
- JSON 解析错误  
- API 响应错误
- 其他未知错误

错误信息会在 ComfyUI 控制台中输出，并返回包含错误信息的 JSON。 