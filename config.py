"""
全局配置模块

该模块定义了LLM部署系统的全局配置参数，包括：
- 主机地址配置
- 健康检查配置
- 进程终止配置
- 模型权重根目录配置
"""

HOST = "127.0.0.1"
PORT = 8000
HEALTH_CHECK_TIMEOUT = 120
HEALTH_CHECK_INTERVAL = 1
PROCESS_TERMINATE_TIMEOUT = 10
MODEL_WEIGHTS_DIR = r"/data2/home/songzhihua/LLM/llm_playground/model_weights"
