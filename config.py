"""
全局配置模块

该模块定义了LLM部署系统的全局配置参数，包括：
- 主机地址配置
- 模型权重根目录配置
- 健康检查配置
- 进程终止配置
- 模型加载超时配置
"""

# 主机地址配置
# HOST = "127.0.0.1"  # 本地运行
HOST = "0.0.0.0"  # docker 内运行 
PORT = 8000

# 模型权重根目录配置
# MODEL_WEIGHTS_DIR = r"WarwickSong/model_weights"  # 本地运行
MODEL_WEIGHTS_DIR = "/app/model_weights"  # docker 内运行

# 最大并发加载数 和 GPU串行加载配置
MAX_CONCURRENT_LOAD = 2
GPU_SERIAL_LOAD = True

# 可用 LLM 和 可用 Embedding 
MODELS = {  
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
}
EMBEDDINGS = {  
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-m3",
}

# 控制最大并发推理请求数，避免GPU显存溢出
# 可根据实际硬件配置和测试结果调整
MAX_CONCURRENT_INFERENCE = 2

# 健康检查配置
HEALTH_CHECK_TIMEOUT = 120
HEALTH_CHECK_INTERVAL = 1
PROCESS_TERMINATE_TIMEOUT = 10

# 模型加载超时配置（秒）
MODEL_LOAD_TIMEOUT = 300  # 模型加载总超时时间
MODEL_STARTUP_TIMEOUT = 300  # 模型进程启动超时时间
MODEL_PORT_READ_TIMEOUT = 300  # 读取端口信息超时时间
