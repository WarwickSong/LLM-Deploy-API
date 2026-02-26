"""
模型进程管理模块

该模块负责管理LLM模型的子进程生命周期，包括：
- 启动模型子进程
- 关闭模型子进程
- 维护模型注册表
- 健康检查
- 进程崩溃检测
"""

import asyncio
import subprocess
import time
import httpx
from config import HOST, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_INTERVAL, PROCESS_TERMINATE_TIMEOUT


class ModelManager:
    """
    模型进程管理器

    负责管理所有加载的模型进程，包括启动、卸载、查询等功能。
    每个模型运行在独立的子进程中，具有独立的端口。

    Attributes:
        registry: 模型注册表，键为模型名称，值为包含进程、端口和路径的字典
        lock: 异步锁，用于并发控制
    """

    def __init__(self):
        """初始化模型管理器，创建空的注册表和锁"""
        self.registry = {}
        self.lock = asyncio.Lock()

    async def load_model(self, model_name: str, model_path: str):
        """
        加载模型到新的子进程中

        Args:
            model_name: 模型名称，格式为"组织/模型名"
            model_path: 模型权重路径

        Raises:
            RuntimeError: 当模型已加载或启动失败时抛出异常
        """
        async with self.lock:
            if model_name in self.registry:
                return

            process = subprocess.Popen(
                [
                    "python",
                    "model_worker.py",
                    "--model_name", model_name,
                    "--model_path", model_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            port = None
            while True:
                line = process.stdout.readline()
                if line.startswith("WORKER_PORT:"):
                    port = int(line.strip().split(":")[1])
                    break

            if port is None:
                raise RuntimeError("Worker 启动失败")

            await self._wait_for_health(port)

            self.registry[model_name] = {
                "process": process,
                "port": port,
                "model_path": model_path,
            }

    async def _wait_for_health(self, port: int):
        """
        等待模型worker健康检查通过

        Args:
            port: 模型worker的端口号

        Raises:
            RuntimeError: 当健康检查超时时抛出异常
        """
        start = time.time()
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    r = await client.get(f"http://{HOST}:{port}/health")
                    if r.status_code == 200:
                        return
                except Exception:
                    pass

                if time.time() - start > HEALTH_CHECK_TIMEOUT:
                    raise RuntimeError("模型启动超时")

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def unload_model(self, model_name: str):
        """
        卸载模型并释放资源

        Args:
            model_name: 要卸载的模型名称
        """
        async with self.lock:
            if model_name not in self.registry:
                return

            process = self.registry[model_name]["process"]

            process.terminate()

            try:
                process.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()

            del self.registry[model_name]

    def list_models(self):
        """
        列出所有已加载的模型

        Returns:
            dict: 包含所有已加载模型信息的字典，包括端口、进程ID和模型路径
        """
        result = {}
        for name, info in self.registry.items():
            process = info["process"]

            if process.poll() is not None:
                continue

            result[name] = {
                "port": info["port"],
                "pid": process.pid,
                "model_path": info["model_path"],
            }

        return result

    async def forward_request(self, model_name: str, path: str, payload: dict):
        """
        转发请求到指定的模型worker

        Args:
            model_name: 模型名称
            path: 请求路径，如"/generate"或"/embedding"
            payload: 请求负载

        Returns:
            dict: 模型worker的响应

        Raises:
            RuntimeError: 当模型未加载或进程已崩溃时抛出异常
        """
        if model_name not in self.registry:
            raise RuntimeError("模型未加载")

        info = self.registry[model_name]

        if info["process"].poll() is not None:
            raise RuntimeError("模型进程已崩溃")

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"http://{HOST}:{info['port']}{path}",
                json=payload,
            )
            return r.json()
