"""
Pydantic数据模型定义模块

该模块定义了API请求和响应的数据模型，包括：
- LoadModelRequest: 模型加载请求
- GenerateRequest: 文本生成请求
- EmbeddingRequest: 文本嵌入请求
"""

from pydantic import BaseModel
from typing import List, Dict, Any


class LoadModelRequest(BaseModel):
    """
    模型加载请求模型

    Attributes:
        model_name: 模型名称，格式为"组织/模型名"，如"Qwen/Qwen3-8B"
    """

    model_name: str


class GenerateRequest(BaseModel):
    """
    文本生成请求模型

    Attributes:
        model_name: 模型名称，格式为"组织/模型名"
        messages: 消息列表，每个消息包含role和content字段
        max_new_tokens: 最大新生成token数，默认为8192
    """

    model_name: str
    messages: List[Dict[str, Any]]
    max_new_tokens: int = 8192


class EmbeddingRequest(BaseModel):
    """
    文本嵌入请求模型

    Attributes:
        model_name: 模型名称，格式为"组织/模型名"
        texts: 待编码的文本列表
        query: 是否为查询文本，默认为False
    """

    model_name: str
    texts: List[str]
    query: bool = False
