"""
模型进程管理模块 V2 - 工业级实现

该模块负责管理LLM模型的子进程生命周期，包括：
- 启动模型子进程
- 关闭模型子进程
- 维护模型注册表
- 健康检查
- 进程崩溃检测
- 启动超时和自动回滚

设计原则：
1. 精细锁粒度：只在 registry 操作时加锁，允许并行加载不同模型
2. 超时机制：每个阶段都有超时保护
3. 异常回滚：启动失败时自动清理资源
4. 同步进程创建：使用 subprocess.Popen 兼容大模型加载
"""

import asyncio
import subprocess
import time
import httpx
import psutil
from config import (
    HOST, 
    MODELS,
    EMBEDDINGS,
    HEALTH_CHECK_TIMEOUT, 
    HEALTH_CHECK_INTERVAL, 
    PROCESS_TERMINATE_TIMEOUT,
    MODEL_LOAD_TIMEOUT,
    MODEL_STARTUP_TIMEOUT,
    MODEL_PORT_READ_TIMEOUT
)

class ModelLoadError(Exception):
    """模型加载异常"""
    pass


class ModelManager:
    """
    模型进程管理器 V2

    负责管理所有加载的模型进程，包括启动、卸载、查询等功能。
    每个模型运行在独立的子进程中，具有独立的端口。

    Attributes:
        registry: 模型注册表，键为模型名称，值为包含进程、端口和路径的字典
        _lock: 异步锁，用于保护 registry 的并发访问
        _loading_set: 正在加载中的模型集合，用于防止重复加载
        _load_semaphore: 加载并发控制信号量
        _gpu_serial_load: GPU 加载串行开关
    """

    def __init__(self, max_concurrent_load=2, gpu_serial_load=False):
        """初始化模型管理器"""
        self.registry = {}
        self._lock = asyncio.Lock()
        self._loading_set = set()
        self._load_semaphore = asyncio.Semaphore(max_concurrent_load)
        self._gpu_serial_load = gpu_serial_load
        self._gpu_load_lock = asyncio.Lock() if gpu_serial_load else None

    async def load_model(self, model_name: str, model_path: str):
        """
        加载模型到新的子进程中（工业级实现）

        设计特点：
        1. 分阶段执行，只在关键步骤加锁
        2. 完全异步的进程管理
        3. 完善的超时和异常处理
        4. 自动资源回滚
        5. 并发控制
        6. GPU 加载串行选项

        Args:
            model_name: 模型名称，格式为"组织/模型名"
            model_path: 模型权重路径

        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        # 阶段1：快速检查（短锁）
        async with self._lock:
            if model_name in self.registry:
                return
            if model_name in self._loading_set:
                raise ModelLoadError(f"模型 {model_name} 正在加载中，请稍后再试")
            self._loading_set.add(model_name)

        process = None
        port = None
        
        try:
            # 阶段1.5：并发控制
            async with self._load_semaphore:
                # 阶段1.6：GPU 加载串行（如果启用）
                if self._gpu_serial_load:
                    async with self._gpu_load_lock:
                        # 阶段2：启动进程（无锁，允许并行）
                        process = await self._start_worker_process(model_name, model_path)
                        
                        # 阶段3：读取端口（无锁，异步等待）
                        port = await self._read_worker_port(process)
                        
                        # 阶段4：健康检查（无锁）
                        await self._wait_for_health(port)
                else:
                    # 阶段2：启动进程（无锁，允许并行）
                    process = await self._start_worker_process(model_name, model_path)
                    
                    # 阶段3：读取端口（无锁，异步等待）
                    port = await self._read_worker_port(process)
                    
                    # 阶段4：健康检查（无锁）
                    await self._wait_for_health(port)
                
                # 阶段5：注册模型（短锁）
                async with self._lock:
                    self.registry[model_name] = {
                        "process": process,
                        "port": port,
                        "model_path": model_path,
                    }
                    
        except Exception as e:
            # 异常回滚：清理已启动的进程
            if process is not None:
                await self._terminate_process(process)
            raise ModelLoadError(f"加载模型 {model_name} 失败: {str(e)}")
            
        finally:
            # 从加载集合中移除
            async with self._lock:
                self._loading_set.discard(model_name)

    async def _start_worker_process(self, model_name: str, model_path: str) -> subprocess.Popen:
        """
        启动模型工作进程（回退为同步创建，兼容大模型加载）

        Args:
            model_name: 模型名称
            model_path: 模型路径

        Returns:
            subprocess.Popen: 启动的进程对象

        Raises:
            ModelLoadError: 当进程启动失败时抛出异常
        """
        try:
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
                bufsize=1,
                universal_newlines=True,
            )
            return process
        except Exception as e:
            raise ModelLoadError(f"启动工作进程失败: {str(e)}")

    async def _read_worker_port(self, process: subprocess.Popen) -> int:
        """
        读取工作进程报告的端口号（带 handshake 机制，适配同步进程）

        设计特点：
        1. 首先等待 MODEL_LOADING_STARTED 信号
        2. 然后读取 WORKER_PORT 信号
        3. 完善的超时和异常处理

        Args:
            process: 工作进程对象

        Returns:
            int: 端口号

        Raises:
            ModelLoadError: 当读取失败或超时时抛出异常
        """
        import select

        start_time = time.time()
        loading_started = False

        while True:
            # 检查是否超时
            if time.time() - start_time > MODEL_PORT_READ_TIMEOUT:
                raise ModelLoadError(f"读取端口信息超时（{MODEL_PORT_READ_TIMEOUT}秒）")

            # 检查进程是否已退出
            if process.poll() is not None:
                stderr = process.stderr.read()[:1000]
                raise ModelLoadError(f"工作进程异常退出（返回码: {process.poll()}）: {stderr}")

            # 同步读取输出（行缓冲，非阻塞检查）
            try:
                # 先检查是否有数据，避免阻塞
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        line_str = line.strip()

                        # 检测加载开始信号
                        if line_str == "MODEL_LOADING_STARTED":
                            loading_started = True
                            continue

                        # 只在收到加载开始信号后才读取端口
                        if loading_started and line_str.startswith("WORKER_PORT:"):
                            try:
                                port = int(line_str.split(":")[1])
                                return port
                            except (IndexError, ValueError):
                                raise ModelLoadError(f"无效的端口信息格式: {line_str}")
            except Exception as e:
                raise ModelLoadError(f"读取端口信息时出错: {str(e)}")

    async def _wait_for_health(self, port: int):
        """
        等待模型worker健康检查通过（带超时）

        Args:
            port: 模型worker的端口号

        Raises:
            ModelLoadError: 当健康检查超时时抛出异常
        """
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while True:
                # 检查总超时
                if time.time() - start_time > MODEL_STARTUP_TIMEOUT:
                    raise ModelLoadError(f"健康检查超时（{MODEL_STARTUP_TIMEOUT}秒）")
                
                try:
                    r = await client.get(
                        f"http://{HOST}:{port}/health",
                        timeout=5.0
                    )
                    if r.status_code == 200:
                        return
                except Exception:
                    pass

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _terminate_process(self, process: subprocess.Popen):
        """
        优雅终止同步进程

        Args:
            process: 要终止的进程
        """
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception:
            pass

    def _kill_process_tree(self, pid: int):
        """
        彻底终止进程及其所有子进程（同步方法，用于卸载）

        Args:
            pid: 主进程ID
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            gone, alive = psutil.wait_procs(children, timeout=3)
            
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            parent.terminate()
            try:
                parent.wait(timeout=3)
            except psutil.TimeoutExpired:
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

    async def unload_model(self, model_name: str, auto_cleanup: bool = True):
        """
        卸载模型并释放资源

        Args:
            model_name: 要卸载的模型名称
            auto_cleanup: 是否自动清理僵尸进程，默认为True

        Returns:
            dict: 卸载结果，包含清理的僵尸进程信息
        """
        async with self._lock:
            if model_name not in self.registry:
                return {
                    "status": "unloaded",
                    "message": "模型未加载",
                    "cleanup_result": {"cleaned_count": 0, "cleaned_pids": []}
                }

            info = self.registry[model_name]
            process = info["process"]

            # 终止模型进程
            await self._terminate_process(process)

            del self.registry[model_name]

            # 自动清理僵尸进程
            cleanup_result = {"cleaned_count": 0, "cleaned_pids": []}
            if auto_cleanup:
                cleanup_result = self.cleanup_zombie_processes()

            return {
                "status": "unloaded",
                "message": f"模型 {model_name} 已卸载",
                "cleanup_result": cleanup_result
            }

    def list_models(self):
        """
        列出所有已加载的模型

        Returns:
            dict: 包含所有已加载模型信息的字典
        """
        result = {}
        for name, info in self.registry.items():
            process = info["process"]

            # 检查进程状态（同步进程使用 poll()）
            if process.poll() is not None:
                continue

            result[name] = {
                "port": info["port"],
                "pid": process.pid if hasattr(process, 'pid') else None,
                "model_path": info["model_path"],
            }

        return result

    async def forward_request(self, model_name: str, path: str, payload: dict):
        """
        转发请求到指定的模型worker

        Args:
            model_name: 模型名称
            path: 请求路径
            payload: 请求负载

        Returns:
            dict: 模型worker的响应

        Raises:
            RuntimeError: 当模型未加载或进程已崩溃时抛出异常
        """
        async with self._lock:
            if model_name not in self.registry:
                raise RuntimeError("模型未加载")

            info = self.registry[model_name]
            process = info["process"]
            
            # 检查进程状态（同步进程使用 poll()）
            if process.poll() is not None:
                raise RuntimeError("模型进程已崩溃")

            port = info["port"]

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"http://{HOST}:{port}{path}",
                json=payload,
            )
            return r.json()

    def cleanup_zombie_processes(self):
        """
        清理所有与模型相关的僵尸进程，但保留正在运行的正常模型进程和Embedding模型的工作进程

        Returns:
            dict: 清理结果，包含清理的进程数量
        """
        cleaned_count = 0
        cleaned_pids = []
        
        # 获取当前注册的正常模型进程PID列表
        normal_pids = set()
        embedding_model_pids = set()
        
        for model_name, info in self.registry.items():
            try:
                process = info["process"]
                # 获取进程ID（同步进程使用 poll() 检查状态）
                if process.poll() is None:
                    normal_pids.add(process.pid)
                    if model_name in EMBEDDINGS:
                        embedding_model_pids.add(process.pid)
            except Exception:
                pass
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'ppid']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 0:
                        cmdline_str = ' '.join(cmdline)
                        pid = proc.info['pid']
                        ppid = proc.info['ppid']
                        
                        # 跳过正常运行的模型进程
                        if pid in normal_pids:
                            continue
                        
                        # 保护Embedding模型的子进程
                        if ppid in embedding_model_pids:
                            continue
                        
                        # 只清理真正的僵尸进程和孤立的multiprocessing.spawn进程
                        is_zombie = False
                        
                        # 检查是否为僵尸进程
                        if proc.info['status'] == psutil.STATUS_ZOMBIE:
                            is_zombie = True
                        # 检查是否为孤立的multiprocessing.spawn进程
                        elif 'multiprocessing.spawn' in cmdline_str:
                            try:
                                parent = proc.parent()
                                if parent is None or not parent.is_running():
                                    is_zombie = True
                                elif parent.pid not in embedding_model_pids:
                                    is_zombie = True
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                is_zombie = True
                        # 检查是否为已退出的model_worker进程
                        elif 'model_worker.py' in cmdline_str:
                            try:
                                parent = proc.parent()
                                if parent is None or not parent.is_running():
                                    is_zombie = True
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                is_zombie = True
                        
                        if is_zombie:
                            try:
                                proc.kill()
                                cleaned_count += 1
                                cleaned_pids.append(pid)
                            except psutil.NoSuchProcess:
                                pass
                            except psutil.AccessDenied:
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception:
            pass
        
        return {"cleaned_count": cleaned_count, "cleaned_pids": cleaned_pids}
