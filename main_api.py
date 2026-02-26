"""
主API服务模块

该模块提供FastAPI服务，作为LLM模型调度的主入口，包括：
- 模型加载和卸载接口
- 文本生成接口
- 文本嵌入接口
"""

import os
from fastapi import FastAPI
from schemas import *
from model_manager import ModelManager
from config import (
    MODEL_WEIGHTS_DIR, 
    HOST, 
    PORT
)

app = FastAPI()
manager = ModelManager()


def get_model_path(model_name: str) -> str:
    """
    根据模型名称获取完整的模型权重路径

    Args:
        model_name: 模型名称，格式为"组织/模型名"，如"Qwen/Qwen3-8B"

    Returns:
        str: 完整的模型权重路径
    """
    return os.path.join(MODEL_WEIGHTS_DIR, model_name.replace("/", os.sep))


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
        dict: 包含卸载状态的响应
    """
    await manager.unload_model(req.model_name)
    return {"status": "unloaded"}


@app.get("/model/list")
async def list_models():
    """
    列出所有已加载的模型

    Returns:
        dict: 包含所有已加载模型信息的字典
    """
    return manager.list_models()


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
    uvicorn.run(app, host=HOST, port=PORT)
