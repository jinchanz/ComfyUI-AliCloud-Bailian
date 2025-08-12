import json
import requests
import time
import asyncio
import aiohttp
from .logging import logger


def _poll_task_result(task_id, api_key, poll_interval, max_wait_time):
    """轮询任务结果"""
    task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }
    
    start_time = time.time()
    
    while True:
        try:
            # 检查是否超时
            if time.time() - start_time > max_wait_time:
                error_msg = f"任务轮询超时 ({max_wait_time}秒)"
                logger.info(f"[BailianAPI] {error_msg}")
                return {"error": error_msg, "task_id": task_id}
            
            logger.info(f"[BailianAPI] 轮询任务状态: {task_id}")
            
            # 查询任务状态
            response = requests.get(task_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            result_data = response.json()
            task_status = result_data.get("output", {}).get("task_status", "")
            
            logger.info(f"[BailianAPI] 任务状态: {task_status}")
            
            # 任务完成
            if task_status == "SUCCEEDED":
                logger.info(f"[BailianAPI] 任务完成成功")
                return result_data
            
            # 任务失败
            elif task_status == "FAILED":
                error_code = result_data.get("output", {}).get("code", "unknown")
                error_message = result_data.get("output", {}).get("message", "任务执行失败")
                logger.info(f"[BailianAPI] 任务执行失败: {error_code} - {error_message}")
                return result_data
            
            # 继续等待
            elif task_status in ["PENDING", "RUNNING"]:
                time.sleep(poll_interval)
                continue
            
            # 未知状态
            else:
                logger.info(f"[BailianAPI] 未知任务状态: {task_status}")
                return result_data
                
        except requests.exceptions.RequestException as e:
            error_msg = f"轮询请求失败: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return {"error": error_msg, "task_id": task_id}
            
        except Exception as e:
            error_msg = f"轮询过程出错: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return {"error": error_msg, "task_id": task_id}

async def _async_poll_task_result(session, task_id, api_key, poll_interval, max_wait_time):
    """异步轮询任务结果"""
    task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }
    
    start_time = time.time()
    
    while True:
        try:
            # 检查是否超时
            if time.time() - start_time > max_wait_time:
                error_msg = f"任务轮询超时 ({max_wait_time}秒)"
                logger.info(f"[BailianAPI] {error_msg}")
                return {"error": error_msg, "task_id": task_id}
            
            logger.info(f"[BailianAPI] 轮询任务状态: {task_id}")
            
            # 查询任务状态
            async with session.get(task_url, headers=headers) as response:
                response.raise_for_status()
                result_data = await response.json()
            
            task_status = result_data.get("output", {}).get("task_status", "")
            logger.info(f"[BailianAPI] 任务状态: {task_status}")
            
            # 任务完成
            if task_status == "SUCCEEDED":
                logger.info(f"[BailianAPI] 任务完成成功")
                return result_data
            
            # 任务失败
            elif task_status == "FAILED":
                error_code = result_data.get("output", {}).get("code", "unknown")
                error_message = result_data.get("output", {}).get("message", "任务执行失败")
                logger.info(f"[BailianAPI] 任务执行失败: {error_code} - {error_message}")
                return result_data
            
            # 继续等待
            elif task_status in ["PENDING", "RUNNING"]:
                await asyncio.sleep(poll_interval)
                continue
            
            # 未知状态
            else:
                logger.info(f"[BailianAPI] 未知任务状态: {task_status}")
                return result_data
                
        except Exception as e:
            error_msg = f"轮询过程出错: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return {"error": error_msg, "task_id": task_id}

async def _async_create_and_poll_refiner_task(session, endpoint, gender, input, result_image_url, api_key, poll_interval, max_wait_time):
    """异步创建和轮询refiner任务"""
    request_data = {
        "model": "aitryon-refiner",
        "input": {
            "coarse_image_url": result_image_url,
            **input
        },
        "parameters": {
            "gender": gender,
        }
    }
    logger.info(f"[VirtualTryOn Refiner] 发送请求到: {endpoint}, 请求数据: {request_data}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "X-DashScope-Async": "enable"
    }
    
    async with session.post(endpoint, headers=headers, json=request_data) as response:
        if response.status != 200:
            response_text = await response.text()
            raise ValueError(f"API 请求失败: {response.status} {response_text}")
        response_data = await response.json()
    
    logger.info(f"[VirtualTryOn] 请求成功: {response_data}")
    
    # 如果是异步模式且有task_id，需要轮询结果
    if "output" in response_data and "task_id" in response_data["output"]:
        task_id = response_data["output"]["task_id"]
        task_status = response_data["output"].get("task_status", "")
        
        logger.info(f"[BailianAPI] 获取到任务ID: {task_id}, 状态: {task_status}")
        
        # 如果任务是PENDING状态，开始轮询
        if task_status == "PENDING":
            response_data = await _async_poll_task_result(session, task_id, api_key, poll_interval, max_wait_time)
            return response_data
        else:
            return response_data
    else:
        return response_data

async def _async_process_single_person(session, person_image, top_garment_image, bottom_garment_image, model, parameters, endpoint, headers, async_mode, enable_refiner, gender, api_key, poll_interval, max_wait_time):
    """异步处理单个人物图像"""
    try:
        input = {
            "top_garment_url": top_garment_image,
            "bottom_garment_url": bottom_garment_image if bottom_garment_image is not None and bottom_garment_image.strip() != "" else "",
            "person_image_url": person_image,
        }
        request_data = {
            "model": model,
            "input": input,
        }
        if parameters is not None and parameters.strip() != "":
            request_data["parameters"] = parameters if isinstance(parameters, dict) else json.loads(parameters)
        else:
            request_data["parameters"] = {}
        
        logger.info(f"[VirtualTryOn] 发送请求到: {endpoint}, 请求数据: {request_data}")
        
        async with session.post(endpoint, headers=headers, json=request_data) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(f"API 请求失败: {response.status} {response_text}")
            response_data = await response.json()
        
        logger.info(f"[VirtualTryOn] 请求成功: {response_data}")
        
        # 如果是异步模式且有task_id，需要轮询结果
        if async_mode and "output" in response_data and "task_id" in response_data["output"]:
            task_id = response_data["output"]["task_id"]
            task_status = response_data["output"].get("task_status", "")
            
            logger.info(f"[BailianAPI] 获取到任务ID: {task_id}, 状态: {task_status}")
            
            # 如果任务是PENDING状态，开始轮询
            if task_status == "PENDING":
                response_data = await _async_poll_task_result(session, task_id, api_key, poll_interval, max_wait_time)
                if enable_refiner:
                    try:
                        response_data = await _async_create_and_poll_refiner_task(session, endpoint, gender, input, response_data["output"]["image_url"], api_key, poll_interval, max_wait_time)
                    except Exception as e:
                        error_msg = f"处理refiner任务失败: {str(e)}"
                        logger.info(f"[VirtualTryOn] {error_msg}")
                        
                return response_data
        else:
            # 同步模式
            if enable_refiner:
                try:
                    response_data = await _async_create_and_poll_refiner_task(session, endpoint, gender, input, response_data["output"]["image_url"], api_key, poll_interval, max_wait_time)
                except Exception as e:
                    error_msg = f"处理refiner任务失败: {str(e)}"
                    logger.info(f"[VirtualTryOn] {error_msg}")
            return response_data
            
    except Exception as e:
        error_msg = f"处理人物图像失败: {str(e)}"
        logger.info(f"[VirtualTryOn] {error_msg}")
        return {"error": error_msg, "person_image": person_image}

def create_and_poll_refiner_task(endpoint, gender, input, result_image_url, api_key, poll_interval, max_wait_time):
    request_data = {
        "model": "aitryon-refiner",
        "input": {
            "coarse_image_url": result_image_url,
            **input
        },
        "parameters": {
            "gender": gender,
        }
    }
    logger.info(f"[VirtualTryOn Refiner] 发送请求到: {endpoint}, 请求数据: {request_data}")
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "X-DashScope-Async": "enable"
    }
    
    response = requests.post(endpoint, headers=headers, json=request_data)
    if response.status_code != 200:
        raise ValueError(f"API 请求失败: {response.status_code} {response.text}")
    response_data = response.json()
    logger.info(f"[VirtualTryOn] 请求成功: {response_data}")
    # 如果是异步模式且有task_id，需要轮询结果
    if "output" in response_data and "task_id" in response_data["output"]:
        task_id = response_data["output"]["task_id"]
        task_status = response_data["output"].get("task_status", "")
        
        logger.info(f"[BailianAPI] 获取到任务ID: {task_id}, 状态: {task_status}")
        
        # 如果任务是PENDING状态，开始轮询
        if task_status == "PENDING":
            response_data = _poll_task_result(task_id, api_key, poll_interval, max_wait_time)
            return response_data
        else:
            return response_data
    else:
        return response_data

class BailianAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {"default": "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis/"}),
                "params": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "aitryon-plus"}),
                "async_mode": ("BOOLEAN", {"default": True}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait_time": ("INT", {"default": 300, "min": 30, "max": 1800}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def run(self, endpoint, params, api_key="", model="aitryon-plus", async_mode=True, poll_interval=3, max_wait_time=300):
        try:
            # 解析输入参数
            input_params = json.loads(params) if isinstance(params, str) else params
            
            # 构建请求数据
            request_data = {
                "model": model,
                "input": input_params.get("input", {}),
                "parameters": input_params.get("parameters", {})
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
            
            # 如果启用异步模式，添加异步头
            if async_mode:
                headers["X-DashScope-Async"] = "enable"
            
            logger.info(f"[BailianAPI] 发送请求到: {endpoint}")
            
            # 发送 POST 请求
            response = requests.post(
                endpoint,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 返回响应内容
            response_data = response.json()
            
            # 如果是异步模式且有task_id，需要轮询结果
            if async_mode and "output" in response_data and "task_id" in response_data["output"]:
                task_id = response_data["output"]["task_id"]
                task_status = response_data["output"].get("task_status", "")
                
                logger.info(f"[BailianAPI] 获取到任务ID: {task_id}, 状态: {task_status}")
                
                # 如果任务是PENDING状态，开始轮询
                if task_status == "PENDING":
                    task_result = _poll_task_result(task_id, api_key, poll_interval, max_wait_time)
                    return (json.dumps(task_result, ensure_ascii=False, indent=2),)
                
            # 直接返回结果（同步模式或已完成的任务）
            return (json.dumps(response_data, ensure_ascii=False, indent=2),)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API 请求失败: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.info(f"[BailianAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

class BailianAPISubmit:
    """提交阿里云百炼API任务，返回task_id"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {"default": "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis/"}),
                "params": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "aitryon-plus"}),
                "async_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_id", "response")

    FUNCTION = "submit"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def submit(self, endpoint, params, api_key="", model="aitryon-plus", async_mode=True):
        try:
            # 解析输入参数
            input_params = json.loads(params) if isinstance(params, str) else params
            
            # 构建请求数据
            request_data = {
                "model": model,
                "input": input_params.get("input", {}),
                "parameters": input_params.get("parameters", {})
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
            
            # 如果启用异步模式，添加异步头
            if async_mode:
                headers["X-DashScope-Async"] = "enable"
            
            logger.info(f"[BailianAPISubmit] 提交任务到: {endpoint}")
            
            # 发送 POST 请求
            response = requests.post(
                endpoint,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 返回响应内容
            response_data = response.json()
            
            # 提取task_id
            task_id = ""
            if "output" in response_data and "task_id" in response_data["output"]:
                task_id = response_data["output"]["task_id"]
                task_status = response_data["output"].get("task_status", "")
                logger.info(f"[BailianAPISubmit] 任务提交成功，ID: {task_id}, 状态: {task_status}")
            else:
                logger.info(f"[BailianAPISubmit] 同步请求完成")
            
            response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
            return (task_id, response_json)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API 请求失败: {str(e)}"
            logger.info(f"[BailianAPISubmit] {error_msg}")
            error_response = json.dumps({"error": error_msg}, ensure_ascii=False)
            return ("", error_response)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            logger.info(f"[BailianAPISubmit] {error_msg}")
            error_response = json.dumps({"error": error_msg}, ensure_ascii=False)
            return ("", error_response)
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.info(f"[BailianAPISubmit] {error_msg}")
            error_response = json.dumps({"error": error_msg}, ensure_ascii=False)
            return ("", error_response)


class BailianAPIPoll:
    """轮询阿里云百炼API任务结果"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_id": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait_time": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "single_query": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "status")

    FUNCTION = "poll"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def poll(self, task_id, api_key="", poll_interval=3, max_wait_time=300, single_query=False):
        if not task_id or task_id.strip() == "":
            error_msg = "task_id 不能为空"
            logger.info(f"[BailianAPIPoll] {error_msg}")
            error_response = json.dumps({"error": error_msg}, ensure_ascii=False)
            return (error_response, "ERROR")
        
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id.strip()}"
        headers = {
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        
        start_time = time.time()
        
        while True:
            try:
                logger.info(f"[BailianAPIPoll] 查询任务状态: {task_id}")
                
                # 查询任务状态
                response = requests.get(task_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                result_data = response.json()
                task_status = result_data.get("output", {}).get("task_status", "")
                
                logger.info(f"[BailianAPIPoll] 任务状态: {task_status}")
                
                # 任务完成
                if task_status == "SUCCEEDED":
                    logger.info(f"[BailianAPIPoll] 任务完成成功")
                    result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                    return (result_json, "SUCCEEDED")
                
                # 任务失败
                elif task_status == "FAILED":
                    error_code = result_data.get("output", {}).get("code", "unknown")
                    error_message = result_data.get("output", {}).get("message", "任务执行失败")
                    logger.info(f"[BailianAPIPoll] 任务执行失败: {error_code} - {error_message}")
                    result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                    return (result_json, "FAILED")
                
                # 如果是单次查询模式，直接返回当前状态
                if single_query:
                    result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                    return (result_json, task_status)
                
                # 继续等待
                elif task_status in ["PENDING", "RUNNING"]:
                    # 检查是否超时
                    if time.time() - start_time > max_wait_time:
                        error_msg = f"任务轮询超时 ({max_wait_time}秒)"
                        logger.info(f"[BailianAPIPoll] {error_msg}")
                        error_response = json.dumps({"error": error_msg, "task_id": task_id, "last_status": task_status}, ensure_ascii=False)
                        return (error_response, "TIMEOUT")
                    
                    time.sleep(poll_interval)
                    continue
                
                # 未知状态
                else:
                    logger.info(f"[BailianAPIPoll] 未知任务状态: {task_status}")
                    result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                    return (result_json, task_status)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"轮询请求失败: {str(e)}"
                logger.info(f"[BailianAPIPoll] {error_msg}")
                error_response = json.dumps({"error": error_msg, "task_id": task_id}, ensure_ascii=False)
                return (error_response, "ERROR")
                
            except Exception as e:
                error_msg = f"轮询过程出错: {str(e)}"
                logger.info(f"[BailianAPIPoll] {error_msg}")
                error_response = json.dumps({"error": error_msg, "task_id": task_id}, ensure_ascii=False)
                return (error_response, "ERROR")


class MaletteJSONExtractor:
    """从JSON中提取嵌套键值的工具节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "key_path": ("STRING", {"default": "output.image_url"}),
            },
            "optional": {
                "default_value": ("STRING", {"default": ""}),
                "return_as_string": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)

    FUNCTION = "extract"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def extract(self, json_input, key_path, default_value="", return_as_string=True):
        try:
            # 解析JSON
            if isinstance(json_input, str):
                data = json.loads(json_input)
            else:
                data = json_input
            
            # 分割键路径
            keys = key_path.strip().split('.')
            
            # 逐层访问数据
            current_data = data
            for key in keys:
                if isinstance(current_data, dict) and key in current_data:
                    current_data = current_data[key]
                elif isinstance(current_data, list) and key.isdigit():
                    # 支持数组索引，如 "items.0.name"
                    index = int(key)
                    if 0 <= index < len(current_data):
                        current_data = current_data[index]
                    else:
                        raise KeyError(f"数组索引 {index} 超出范围")
                else:
                    raise KeyError(f"键 '{key}' 不存在")
            
            # 处理返回值
            if return_as_string:
                if isinstance(current_data, (dict, list)):
                    result = json.dumps(current_data, ensure_ascii=False, indent=2)
                else:
                    result = str(current_data)
            else:
                result = str(current_data) if current_data is not None else ""
            
            logger.info(f"[JSONExtractor] 成功提取键路径 '{key_path}': {result[:100]}{'...' if len(str(result)) > 100 else ''}")
            formatted_result = None
            try:
                formatted_result = json.loads(result) if isinstance(result, str) else result
            except json.JSONDecodeError:
                logger.info(f"[JSONExtractor] 格式化结果失败，无法解析JSON，可能不是JSON格式: {result}")
                formatted_result = result
            return {"ui": {"json": [formatted_result], "text": [result]}, "result": (formatted_result,)}
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            logger.info(f"[JSONExtractor] {error_msg}")
            return (default_value,)
            
        except KeyError as e:
            error_msg = f"键路径 '{key_path}' 不存在: {str(e)}"
            logger.info(f"[JSONExtractor] {error_msg}")
            return (default_value,)
            
        except Exception as e:
            error_msg = f"提取过程出错: {str(e)}"
            logger.info(f"[JSONExtractor] {error_msg}")
            return (default_value,)


class MaletteJSONModifier:
    """修改JSON中嵌套键值的工具节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "key_path": ("STRING", {"default": "parameters.restore_face"}),
                "new_value": ("STRING", {"default": "false"}),
            },
            "optional": {
                "value_type": (["auto", "string", "number", "boolean", "json"], {"default": "auto"}),
                "create_path": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_json",)

    FUNCTION = "modify"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def modify(self, json_input, key_path, new_value, value_type="auto", create_path=True):
        try:
            # 解析JSON
            if isinstance(json_input, str):
                data = json.loads(json_input)
            else:
                data = json_input.copy() if isinstance(json_input, dict) else json.loads(str(json_input))
            
            # 分割键路径
            keys = key_path.strip().split('.')
            if not keys or keys == ['']:
                raise ValueError("键路径不能为空")
            
            # 转换新值到合适的类型
            converted_value = self._convert_value(new_value, value_type)
            
            # 递归设置值
            self._set_nested_value(data, keys, converted_value, create_path)
            
            # 返回修改后的JSON
            result_json = json.dumps(data, ensure_ascii=False, indent=2)
            logger.info(f"[JSONModifier] 成功修改键路径 '{key_path}' 为: {converted_value}")
            return (result_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            logger.info(f"[JSONModifier] {error_msg}")
            return (json_input,)
            
        except Exception as e:
            error_msg = f"修改过程出错: {str(e)}"
            logger.info(f"[JSONModifier] {error_msg}")
            return (json_input,)

    def _convert_value(self, value_str, value_type):
        """将字符串值转换为指定类型"""
        if value_type == "string":
            return str(value_str)
        elif value_type == "number":
            try:
                # 尝试转换为整数
                if '.' not in value_str and 'e' not in value_str.lower():
                    return int(value_str)
                else:
                    return float(value_str)
            except ValueError:
                raise ValueError(f"无法将 '{value_str}' 转换为数字")
        elif value_type == "boolean":
            if value_str.lower() in ['true', '1', 'yes', 'on']:
                return True
            elif value_str.lower() in ['false', '0', 'no', 'off']:
                return False
            else:
                raise ValueError(f"无法将 '{value_str}' 转换为布尔值")
        elif value_type == "json":
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                raise ValueError(f"无法将 '{value_str}' 解析为JSON")
        else:  # auto
            # 自动检测类型
            value_str = value_str.strip()
            
            # 检查布尔值
            if value_str.lower() in ['true', 'false']:
                return value_str.lower() == 'true'
            
            # 检查null
            if value_str.lower() == 'null':
                return None
            
            # 检查数字
            try:
                if '.' in value_str or 'e' in value_str.lower():
                    return float(value_str)
                else:
                    return int(value_str)
            except ValueError:
                pass
            
            # 检查JSON对象或数组
            if value_str.startswith(('{', '[')):
                try:
                    return json.loads(value_str)
                except json.JSONDecodeError:
                    pass
            
            # 默认返回字符串
            return value_str

    def _set_nested_value(self, data, keys, value, create_path):
        """递归设置嵌套值"""
        current = data
        
        # 处理除最后一个键之外的所有键
        for i, key in enumerate(keys[:-1]):
            if isinstance(current, dict):
                if key not in current:
                    if create_path:
                        # 检查下一个键是否是数字（用于创建数组）
                        next_key = keys[i + 1]
                        if next_key.isdigit():
                            current[key] = []
                        else:
                            current[key] = {}
                    else:
                        raise KeyError(f"键 '{key}' 不存在，且未启用路径创建")
                current = current[key]
            elif isinstance(current, list):
                if key.isdigit():
                    index = int(key)
                    # 扩展数组到所需长度
                    while len(current) <= index:
                        current.append({})
                    current = current[index]
                else:
                    raise TypeError(f"数组索引必须是数字，得到: '{key}'")
            else:
                raise TypeError(f"无法在类型 {type(current)} 上设置键 '{key}'")
        
        # 设置最后一个键的值
        final_key = keys[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            if final_key.isdigit():
                index = int(final_key)
                # 扩展数组到所需长度
                while len(current) <= index:
                    current.append(None)
                current[index] = value
            else:
                raise TypeError(f"数组索引必须是数字，得到: '{final_key}'")
        else:
            raise TypeError(f"无法在类型 {type(current)} 上设置键 '{final_key}'")

class VirtualTryOn:
    """虚拟试穿"""
    
    """
    {
        "model": "aitryon-plus",
        "input": {
            "top_garment_url": "https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/2389646171/p801332.jpeg",
            "bottom_garment_url": "",
            "person_image_url": "https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1389646171/p801328.png"
        },
        "parameters": {
            "resolution": -1,
            "restore_face": true
        }
    }
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "top_garment_image": ("STRING",),
                "bottom_garment_image": ("STRING",),
                "person_images": ("STRING",),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "endpoint": ("STRING", {"default": "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis/"}),
                "model": ("STRING", {"default": "aitryon-plus"}),
                "parameters": ("STRING", {"default": "{}"}),
                "async_mode": ("BOOLEAN", {"default": True}),
                "enable_refiner": ("BOOLEAN", {"default": False}),
                "gender": ("STRING", {"default": "male", "choices": ["male", "female"]}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "max_wait_time": ("INT", {"default": 300, "min": 30, "max": 1800}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    
    FUNCTION = "run"
    
    
    CATEGORY = "Malette"
    
    async def _async_process_all_persons(self, person_images, top_garment_image, bottom_garment_image, model, parameters, endpoint, headers, async_mode, enable_refiner, gender, api_key, poll_interval, max_wait_time):
        """异步并行处理所有人物图像"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            # 创建所有异步任务
            tasks = [
                _async_process_single_person(
                    session, person_image, top_garment_image, bottom_garment_image,
                    model, parameters, endpoint, headers, async_mode, enable_refiner,
                    gender, api_key, poll_interval, max_wait_time
                )
                for person_image in person_images
            ]
            
            # 并行执行所有任务
            logger.info(f"[VirtualTryOn] 开始并行处理 {len(tasks)} 个人物图像")
            response_data_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(response_data_list):
                if isinstance(result, Exception):
                    error_msg = f"处理第 {i+1} 个人物图像时发生异常: {str(result)}"
                    logger.info(f"[VirtualTryOn] {error_msg}")
                    processed_results.append({"error": error_msg, "person_image": person_images[i]})
                else:
                    processed_results.append(result)
            
            logger.info(f"[VirtualTryOn] 并行处理完成，成功处理 {len([r for r in processed_results if 'error' not in r])} 个，失败 {len([r for r in processed_results if 'error' in r])} 个")
            return processed_results
    
    def _sync_process_all_persons(self, person_images, top_garment_image, bottom_garment_image, model, parameters, endpoint, headers, async_mode, enable_refiner, gender, api_key, poll_interval, max_wait_time):
        """同步处理所有人物图像"""
        logger.info(f"[VirtualTryOn] 开始同步处理 {len(person_images)} 个人物图像")
        processed_results = []
        
        for i, person_image in enumerate(person_images):
            try:
                logger.info(f"[VirtualTryOn] 处理第 {i+1}/{len(person_images)} 个人物图像")
                
                # 构建请求数据
                input_data = {
                    "top_garment_url": top_garment_image,
                    "bottom_garment_url": bottom_garment_image if bottom_garment_image is not None and bottom_garment_image.strip() != "" else "",
                    "person_image_url": person_image,
                }
                
                request_data = {
                    "model": model,
                    "input": input_data,
                }
                
                if parameters is not None and parameters.strip() != "":
                    request_data["parameters"] = parameters if isinstance(parameters, dict) else json.loads(parameters)
                else:
                    request_data["parameters"] = {}
                
                logger.info(f"[VirtualTryOn] 发送请求到: {endpoint}, 请求数据: {request_data}")
                
                # 发送请求
                response = requests.post(endpoint, headers=headers, json=request_data, timeout=300)
                if response.status_code != 200:
                    response_text = response.text
                    raise ValueError(f"API 请求失败: {response.status_code} {response_text}")
                
                response_data = response.json()
                logger.info(f"[VirtualTryOn] 请求成功: {response_data}")
                
                # 如果是异步模式且有task_id，需要轮询结果
                if async_mode and "output" in response_data and "task_id" in response_data["output"]:
                    task_id = response_data["output"]["task_id"]
                    task_status = response_data["output"].get("task_status", "")
                    
                    logger.info(f"[VirtualTryOn] 获取到任务ID: {task_id}, 状态: {task_status}")
                    
                    # 如果任务是PENDING状态，开始轮询
                    if task_status == "PENDING":
                        response_data = _poll_task_result(task_id, api_key, poll_interval, max_wait_time)
                        if enable_refiner:
                            try:
                                response_data = create_and_poll_refiner_task(endpoint, gender, input_data, response_data["output"]["image_url"], api_key, poll_interval, max_wait_time)
                            except Exception as e:
                                error_msg = f"处理refiner任务失败: {str(e)}"
                                logger.info(f"[VirtualTryOn] {error_msg}")
                        
                        processed_results.append(response_data)
                    else:
                        processed_results.append(response_data)
                else:
                    # 同步模式
                    if enable_refiner:
                        try:
                            response_data = create_and_poll_refiner_task(endpoint, gender, input_data, response_data["output"]["image_url"], api_key, poll_interval, max_wait_time)
                        except Exception as e:
                            error_msg = f"处理refiner任务失败: {str(e)}"
                            logger.info(f"[VirtualTryOn] {error_msg}")
                    
                    processed_results.append(response_data)
                    
            except Exception as e:
                error_msg = f"处理第 {i+1} 个人物图像失败: {str(e)}"
                logger.info(f"[VirtualTryOn] {error_msg}")
                processed_results.append({"error": error_msg, "person_image": person_image})
        
        logger.info(f"[VirtualTryOn] 同步处理完成，成功处理 {len([r for r in processed_results if 'error' not in r])} 个，失败 {len([r for r in processed_results if 'error' in r])} 个")
        return processed_results
    
    def run(self, top_garment_image, bottom_garment_image, person_images, api_key, endpoint, model, parameters, async_mode=True, enable_refiner=False, gender="male", poll_interval=3, max_wait_time=300):
        try:
            if not person_images or len(person_images) == 0:
                raise ValueError("person_images 不能为空")
            
            if (not top_garment_image) and (not bottom_garment_image):
                raise ValueError("top_garment_image 和 bottom_garment_image 不能同时为空")

            # 构建请求数据
            person_images = json.loads(person_images)

            if person_images is None or len(person_images) == 0:
                raise ValueError("person_images 不能为空")

            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
            
            # 如果启用异步模式，添加异步头
            if async_mode:
                headers["X-DashScope-Async"] = "enable"

            # 检查是否已经在事件循环中运行
            try:
                # 尝试获取当前事件循环
                loop = asyncio.get_running_loop()
                # 如果已经在事件循环中，使用同步方式处理
                logger.info(f"[VirtualTryOn] 检测到运行中的事件循环，使用同步处理方式")
                response_data_list = self._sync_process_all_persons(
                    person_images, top_garment_image, bottom_garment_image, model, 
                    parameters, endpoint, headers, async_mode, enable_refiner, 
                    gender, api_key, poll_interval, max_wait_time
                )
            except RuntimeError:
                # 没有运行中的事件循环，可以使用 asyncio.run()
                logger.info(f"[VirtualTryOn] 使用异步处理方式")
                response_data_list = asyncio.run(self._async_process_all_persons(
                    person_images, top_garment_image, bottom_garment_image, model, 
                    parameters, endpoint, headers, async_mode, enable_refiner, 
                    gender, api_key, poll_interval, max_wait_time
                ))
                
            return (json.dumps(response_data_list, ensure_ascii=False),)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API 请求失败: {str(e)}"
            logger.info(f"[VirtualTryOn] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            logger.info(f"[VirtualTryOn] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.info(f"[VirtualTryOn] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
    

# 节点映射
NODE_CLASS_MAPPINGS = {
    "BailianAPI": BailianAPI,
    "BailianAPISubmit": BailianAPISubmit,
    "BailianAPIPoll": BailianAPIPoll,
    "MaletteJSONExtractor": MaletteJSONExtractor,
    "MaletteJSONModifier": MaletteJSONModifier,
    "MaletteVirtualTryOn": VirtualTryOn
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BailianAPI": "AliCloud Bailian API",
    "BailianAPISubmit": "AliCloud Bailian API Submit",
    "BailianAPIPoll": "AliCloud Bailian API Poll",
    "MaletteJSONExtractor": "Malette JSON Extractor",
    "MaletteJSONModifier": "Malette JSON Modifier",
    "MaletteVirtualTryOn": "Malette Virtual TryOn"
}