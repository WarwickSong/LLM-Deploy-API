"""
主API服务模块

该模块提供FastAPI服务，作为LLM模型调度的主入口，包括：
- 模型加载和卸载接口
- 文本生成接口
- 文本嵌入接口
- 优雅退出机制
"""

import os
import signal
import sys
import asyncio
from fastapi import FastAPI
from schemas import *
from model_manager import ModelManager
from config import (
    MODEL_WEIGHTS_DIR, 
    HOST, 
    PORT,
    MAX_CONCURRENT_LOAD,
    GPU_SERIAL_LOAD,
)

app = FastAPI()
# 初始化模型管理器，启用 GPU 加载串行以避免显存碎片化
manager = ModelManager(max_concurrent_load=MAX_CONCURRENT_LOAD, gpu_serial_load=GPU_SERIAL_LOAD)
shutdown_event = asyncio.Event()


def get_model_path(model_name: str) -> str:
    """
    根据模型名称获取完整的模型权重路径

    Args:
        model_name: 模型名称，格式为"组织/模型名"，如"Qwen/Qwen3-8B"

    Returns:
        str: 完整的模型权重路径
    """
    return os.path.join(MODEL_WEIGHTS_DIR, model_name.replace("/", os.sep))


async def cleanup_all_models():
    """
    清理所有已加载的模型

    在服务退出前调用，确保所有模型进程被正确终止
    """
    print("\n正在清理所有模型...")
    
    models_to_unload = list(manager.registry.keys())
    for model_name in models_to_unload:
        try:
            result = await manager.unload_model(model_name, auto_cleanup=False)
            print(f"  ✅ 模型 {model_name} 已卸载")
        except Exception as e:
            print(f"  ❌ 卸载模型 {model_name} 时出错: {e}")
    
    print("所有模型清理完成")


def signal_handler(signum, frame):
    """
    信号处理器，用于优雅退出

    在收到SIGTERM或SIGINT信号时，先清理所有模型再退出
    """
    print(f"\n收到退出信号: {signum}")
    shutdown_event.set()
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@app.post("/model/load")
async def load_model(req: LoadModelRequest):
    """
    加载指定的模型

    Args:
        req: 包含模型名称的请求对象

    Returns:
        dict: 包含加载状态的响应
    """
    model_path = get_model_path(req.model_name)
    await manager.load_model(req.model_name, model_path)
    return {"status": "loaded"}


@app.post("/model/unload")
async def unload_model(req: LoadModelRequest):
    """
    卸载指定的模型

    Args:
        req: 包含模型名称的请求对象

    Returns:
        dict: 包含卸载状态和清理结果的响应
    """
    result = await manager.unload_model(req.model_name)
    return result


@app.get("/model/list")
async def list_models():
    """
    列出所有已加载的模型

    Returns:
        dict: 包含所有已加载模型信息的字典
    """
    return manager.list_models()


@app.post("/model/cleanup")
async def cleanup_zombie_processes():
    """
    清理所有与模型相关的僵尸进程

    Returns:
        dict: 清理结果，包含清理的进程数量
    """
    result = manager.cleanup_zombie_processes()
    return result


@app.post("/shutdown")
async def shutdown():
    """
    优雅关闭服务

    清理所有已加载的模型后关闭服务

    Returns:
        dict: 关闭状态
    """
    await cleanup_all_models()
    return {"status": "shutdown", "message": "服务已优雅关闭"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    使用指定模型生成文本

    如果模型未加载，会自动加载模型。

    Args:
        req: 包含模型名称、消息列表和最大生成token数的请求对象

    Returns:
        dict: 包含思考内容和生成响应的字典
    """
    if req.model_name not in manager.registry:
        model_path = get_model_path(req.model_name)
        await manager.load_model(req.model_name, model_path)

    result = await manager.forward_request(
        req.model_name,
        "/generate",
        req.model_dump(),
    )
    return result


@app.post("/embedding")
async def embedding(req: EmbeddingRequest):
    """
    使用指定模型获取文本嵌入

    如果模型未加载，会自动加载模型。

    Args:
        req: 包含模型名称、文本列表和查询标志的请求对象

    Returns:
        dict: 包含文本嵌入向量的字典
    """
    if req.model_name not in manager.registry:
        model_path = get_model_path(req.model_name)
        await manager.load_model(req.model_name, model_path)

    result = await manager.forward_request(
        req.model_name,
        "/embedding",
        req.model_dump(),
    )
    return result

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("LLM部署系统 - 主服务")
    print(f"服务地址: http://{HOST}:{PORT}")
    print("按 Ctrl+C 优雅退出服务")
    print("=" * 60)
    
    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        print("\n\n正在优雅退出...")
        asyncio.run(cleanup_all_models())
        print("服务已停止")
    except Exception as e:
        print(f"\n服务异常退出: {e}")
        asyncio.run(cleanup_all_models())
        raise
