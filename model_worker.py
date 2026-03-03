"""
模型Worker服务模块

该模块作为子进程运行，提供单个模型的推理服务，包括：
- 健康检查接口
- 文本生成接口
- 文本嵌入接口
- 自动端口分配
"""

import argparse
import asyncio
import socket
import uvicorn
import torch
import signal
import sys
from fastapi import FastAPI
from schemas import GenerateRequest, EmbeddingRequest
from models import (
    load_model_and_processor,
    load_embedding_model,
    generate_response,
    get_embedding,
    MODEL2LOADER,
    EMBEDDING2LOADER,
)
from config import(
    MAX_CONCURRENT_INFERENCE
)

app = FastAPI()
model = None
processor = None
embedding_model = None

# 控制最大并发推理请求数，避免GPU显存溢出
# 可根据实际硬件配置和测试结果调整
model_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


def cleanup_resources():
    """
    清理GPU资源和模型对象
    """
    global model, processor, embedding_model
    
    try:
        if model is not None:
            del model
            model = None
    except Exception:
        pass
    
    try:
        if processor is not None:
            del processor
            processor = None
    except Exception:
        pass
    
    try:
        if embedding_model is not None:
            del embedding_model
            embedding_model = None
    except Exception:
        pass
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def signal_handler(signum, frame):
    """
    信号处理器，用于优雅退出
    """
    cleanup_resources()
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@app.get("/health")
async def health():
    """
    健康检查接口

    Returns:
        dict: 包含健康状态的字典
    """
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    生成文本响应

    Args:
        req: 包含模型名称、消息列表和最大生成token数的请求对象

    Returns:
        dict: 包含思考内容和生成响应的字典
    """
    async with model_semaphore:
        thinking, response = generate_response(
            model,
            processor,
            req.messages,
            max_new_tokens=req.max_new_tokens,
        )
    return {"thinking": thinking, "response": response}


@app.post("/embedding")
async def embedding(req: EmbeddingRequest):
    """
    获取文本嵌入向量

    Args:
        req: 包含模型名称、文本列表和查询标志的请求对象

    Returns:
        dict: 包含文本嵌入向量的字典
    """
    async with model_semaphore:
        result = get_embedding(
            embedding_model,
            req.texts,
            query=req.query,
        )
    return {"embedding": result.tolist()}


def find_free_port():
    """
    让操作系统分配可用端口

    Returns:
        int: 操作系统分配的可用端口号
    """
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    """
    主函数，初始化模型并启动服务

    根据模型名称加载相应的模型和处理器，自动分配端口并启动uvicorn服务。
    """
    global model, processor, embedding_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--model_path")
    args = parser.parse_args()

    try:
        # 立即发送加载开始信号
        print("MODEL_LOADING_STARTED", flush=True)

        # 根据模型类型加载
        if args.model_name in EMBEDDING2LOADER:  # 嵌入模型
            embedding_model = load_embedding_model(
                args.model_name, args.model_path
            )
        elif args.model_name in MODEL2LOADER:  # 生成模型  
            model, processor = load_model_and_processor(
                args.model_name, args.model_path
            )
        else:
            raise ValueError(f"不支持的模型: {args.model_name}")

        port = find_free_port()

        print(f"WORKER_PORT:{port}", flush=True)

        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        # 打印错误信息到 stderr，这样父进程可以读取
        import traceback
        error_msg = f"模型加载失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        # 退出进程，返回非零状态码
        sys.exit(1)


if __name__ == "__main__":
    main()
