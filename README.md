# LLM部署系统

## 项目介绍

本系统是一个基于 FastAPI 的多模型独立进程调度平台，用于部署和管理大型语言模型（LLM）和嵌入模型。

### 主要特点

- 每个模型独立进程运行，互不干扰
- 模型卸载即释放显存
- 支持自动加载模型
- 支持并发访问
- 单模型串行保护
- 可扩展为分布式部署
- 自动端口分配，避免端口冲突
- 健康检查机制，确保服务可靠性
- 进程崩溃检测和自动恢复
- 异步处理，提高并发性能
- 优雅退出机制，自动清理资源
- 智能僵尸进程清理，保护Embedding模型
- 加载并发控制，限制同时加载操作数
- GPU 加载串行，避免显存碎片化
- Worker 启动 handshake 机制，确保稳定启动
- 完善的异常回滚，确保资源安全

---

## 架构说明

```
Client
   ↓
Main API (主调度服务)
   ↓ HTTP
Model Worker (独立进程)
   ↓
GPU 推理
```

### 文件结构

```
llm_deploy_api/
│
├── main_api.py              # 主调度服务
├── model_manager.py         # 模型进程管理
├── model_worker.py          # 子进程服务
├── models.py                # 模型加载和推理逻辑
│
├── schemas.py               # Pydantic数据模型
├── config.py                # 全局配置
│
├── docker_build/            # Docker构建配置
│   ├── Dockerfile           # Docker镜像构建文件
│   ├── requirements.txt     # Python依赖包列表
│   ├── start_llm_api.sh     # 启动脚本
│   └── README.md            # Docker部署指南
│
├── test/                    # 测试相关文件
│   ├── test_script.py       # 测试脚本
│   └── test_report/         # 测试报告
│
└── README.md
```

---

## 环境要求

- Python 3.10+
- PyTorch
- FastAPI
- uvicorn
- transformers
- FlagEmbedding
- numpy
- httpx
- psutil

---

## 安装依赖

```bash
pip install fastapi uvicorn transformers torch flagembedding numpy httpx psutil
```

---

## 配置说明

在启动服务前，请先在 `config.py` 中配置以下参数：

```python
# 主机地址
HOST = "127.0.0.1"

# 服务端口
PORT = 8000

# 模型权重根目录
MODEL_WEIGHTS_DIR = r"/path/to/model_weights"

# 最大并发加载数
MAX_CONCURRENT_LOAD = 2  # 限制同时进行的模型加载操作数量

# GPU串行加载配置
GPU_SERIAL_LOAD = True  
# 是否启用 GPU 加载串行
#   - `True`：完全串行加载模型，避免显存碎片化
#   - `False`：并行加载模型，适合多 GPU 环境

# 健康检查配置
HEALTH_CHECK_TIMEOUT = 120  # 健康检查超时时间（秒）
HEALTH_CHECK_INTERVAL = 1    # 健康检查间隔（秒）

# 进程终止配置
PROCESS_TERMINATE_TIMEOUT = 10  # 进程终止超时时间（秒）

# 模型加载超时配置（秒）
MODEL_LOAD_TIMEOUT = 300      # 模型加载总超时时间
MODEL_STARTUP_TIMEOUT = 60    # 模型进程启动超时时间
MODEL_PORT_READ_TIMEOUT = 30  # 读取端口信息超时时间
```

---

## 快速开始

### 模型目录结构

模型权重应按照以下目录结构存放：

```
MODEL_WEIGHTS_DIR/
├── Qwen/
│   ├── Qwen3-8B/
│   ├── Qwen3-30B-A3B-Instruct-2507/
│   ├── Qwen3-VL-8B-Thinking/
│   ├── Qwen3-VL-32B-Instruct/
│   └── Qwen3-VL-30B-A3B-Instruct/
└── BAAI/
    ├── bge-large-zh-v1.5/
    └── bge-m3/
```

### 模型配置

如果可加载的模型权重有增加，需要同时在 `config.py` 和 `models.py` 中添加对应的模型配置：

**1. 在 `config.py` 中添加模型名称到可用集合：**

```python
# 可用 LLM 模型集合
MODELS = {
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
}
# 可用 Embedding 模型集合
EMBEDDINGS = {
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-m3",
}
```

**2. 在 `models.py` 中添加模型名称到加载器的映射：**

```python
# 模型名称到加载器的映射
MODEL2LOADER = {
    "Qwen/Qwen3-8B": Qwen3ForCausalLM,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": Qwen3MoeForCausalLM,
    "Qwen/Qwen3-VL-8B-Thinking": Qwen3VLForConditionalGeneration,
    "Qwen/Qwen3-VL-32B-Instruct": Qwen3VLForConditionalGeneration,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": Qwen3VLMoeForConditionalGeneration,
}
# 嵌入模型名称到加载器的映射
EMBEDDING2LOADER = {
    "BAAI/bge-large-zh-v1.5": FlagModel,
    "BAAI/bge-m3": FlagModel,
}
```

---

## 支持的模型

### LLM模型

| 模型名称 | 说明 | 模型类型 |
|---------|------|---------|
| `Qwen/Qwen3-8B` | Qwen3 8B参数模型 | 文本生成 |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Qwen3 30B MoE指令模型 | 文本生成 |
| `Qwen/Qwen3-VL-8B-Thinking` | Qwen3-VL 8B思考模式模型 | 视觉语言 |
| `Qwen/Qwen3-VL-32B-Instruct` | Qwen3-VL 32B指令模型 | 视觉语言 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | Qwen3-VL 30B MoE指令模型 | 视觉语言 |

### Embedding模型

| 模型名称 | 说明 | 向量维度 |
|---------|------|---------|
| `BAAI/bge-large-zh-v1.5` | BGE大型中文嵌入模型v1.5 | 1024 |
| `BAAI/bge-m3` | BGE多语言嵌入模型 | 1024 |

---

## 使用指南

### 1. 启动主服务

```bash
python main_api.py
```

主服务默认运行在 `http://127.0.0.1:8000`，可以根据需要修改 `config.py` 中的 `HOST` 和 `PORT`。

**启动信息**：
```
============================================================
LLM部署系统 - 主服务
服务地址: http://127.0.0.1:8000
按 Ctrl+C 优雅退出服务
============================================================
```

**优雅退出**：
- 按 `Ctrl+C` 可优雅退出服务
- 退出前会自动清理所有已加载的模型
- 确保GPU内存和进程被正确释放

### 2. 加载模型 / 卸载模型

#### 手动加载模型

```bash
POST http://127.0.0.1:8000/model/load
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-8B",
}
```

系统会自动从config.py中的MODEL_WEIGHTS_DIR拼接路径加载模型，例如，当 `model_name` 为 `"Qwen/Qwen3-8B"` 时，系统会自动加载以下路径的模型：
```bash
${MODEL_WEIGHTS_DIR}/Qwen/Qwen3-8B
```

#### 列出已加载的模型

```bash
GET http://127.0.0.1:8000/model/list
```

#### 卸载模型

```bash
POST http://127.0.0.1:8000/model/unload
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-8B",
}
```

**响应示例**：
```json
{
  "status": "unloaded",
  "message": "模型 Qwen/Qwen3-8B 已卸载",
  "cleanup_result": {
    "cleaned_count": 2,
    "cleaned_pids": [12345, 12346]
  }
}
```

**说明**：
- 模型卸载时会自动清理相关的僵尸进程
- `cleanup_result` 字段显示清理的进程信息
- 如果没有僵尸进程，`cleaned_count` 为 0

### 3. 清理进程 / 关闭服务

#### 清理僵尸进程

```bash
POST http://127.0.0.1:8000/model/cleanup
```

**响应示例**：
```json
{
  "cleaned_count": 3,
  "cleaned_pids": [12345, 12346, 12347]
}
```

#### 优雅关闭服务

```bash
POST http://127.0.0.1:8000/shutdown
```

**响应示例**：
```json
{
  "status": "shutdown",
  "message": "服务已优雅关闭"
}
```

**说明**：
- 调用此接口会清理所有已加载的模型
- 然后优雅关闭服务
- 适用于远程管理或自动化部署场景

### 4. 文本生成

```bash
POST http://127.0.0.1:8000/generate
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-8B",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己。"
    }
  ],
  "max_new_tokens": 8192
}
```

**响应示例**：

```json
{
  "thinking": ["思考内容..."],
  "response": ["我是Qwen3-8B语言模型..."]
}
```

### 5. 多模态生成（Qwen3-VL）

```bash
POST http://127.0.0.1:8000/generate
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-VL-8B-Thinking",
  "messages": [
    {
      "role": "user",
      "content": [
            {
                "type": "image", 
                "image": "base64-encoded-image-string"
            },
            {
                "type": "text", 
                "text": "你好，请描述一下这张图片。"
            }
        ]
    }
  ],
  "max_new_tokens": 8192
}
```

**响应示例**：

```json
{
  "thinking": ["思考内容..."],
  "response": ["这张图片是一张人在沙滩上的照片。"]
}
```

### 6. 文本嵌入

```bash
POST http://127.0.0.1:8000/embedding
Content-Type: application/json

{
  "model_name": "BAAI/bge-large-zh-v1.5",
  "texts": ["这是一个测试句子", "这是另一个测试句子"],
  "query": false
}
```

**响应示例**：

```json
{
  "embedding": [[0.123, 0.456, ...], [0.789, 0.012, ...]]
}
```

**参数说明**：
- `query`: 设置为 `true` 时使用查询编码模式，设置为 `false` 时使用文档编码模式

---

## Python客户端示例

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 加载模型
def load_model(model_name):
    response = requests.post(
        f"{BASE_URL}/model/load",
        json={"model_name": model_name}
    )
    return response.json()

# 文本生成
def generate_text(model_name, messages):
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "model_name": model_name,
            "messages": messages,
            "max_new_tokens": 8192
        }
    )
    return response.json()

# 文本嵌入
def get_embedding(model_name, texts, query=False):
    response = requests.post(
        f"{BASE_URL}/embedding",
        json={
            "model_name": model_name,
            "texts": texts,
            "query": query
        }
    )
    return response.json()

# 模型卸载
def unload_model(model_name):
    response = requests.post(
        f"{BASE_URL}/model/unload",
        json={"model_name": model_name}
    )
    return response.json()

# 清理僵尸进程
def cleanup_zombie_processes():
    response = requests.post(f"{BASE_URL}/model/cleanup")
    return response.json()

# 优雅关闭服务
def shutdown_service():
    response = requests.post(f"{BASE_URL}/shutdown")
    return response.json()

# 使用示例
load_model("Qwen/Qwen3-8B")

result = generate_text(
    "Qwen/Qwen3-8B",
    [{"role": "user", "content": "你好"}]
)
print(result["response"])

embedding = get_embedding(
    "BAAI/bge-large-zh-v1.5",
    ["测试文本"],
    query=True
)
print(embedding["embedding"])

# 卸载模型
unload_model("Qwen/Qwen3-8B")

# 清理僵尸进程
cleanup_result = cleanup_zombie_processes()
print(f"清理了 {cleanup_result['cleaned_count']} 个僵尸进程")

# 优雅关闭服务
shutdown_result = shutdown_service()
print(shutdown_result["message"])
```

---

## 僵尸进程清理

### 问题背景

系统运行一段时间后，可能会出现大量 `python -c "from multiprocessing.spawn import xxx"` 进程，这些进程会持续占用GPU内存，即使卸载了所有模型后依然存在。

### 解决方案

#### 方法1：通过API清理

```bash
curl -X POST http://127.0.0.1:8000/model/cleanup
```

#### 方法2：模型卸载时自动清理（推荐）

系统会在每次模型卸载时自动清理僵尸进程：

```bash
curl -X POST http://127.0.0.1:8000/model/unload \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-8B"}'
```

响应中会包含清理结果：
```json
{
  "status": "unloaded",
  "message": "模型 Qwen/Qwen3-8B 已卸载",
  "cleanup_result": {
    "cleaned_count": 2,
    "cleaned_pids": [12345, 12346]
  }
}
```

#### 方法3：定期清理

建议定期调用清理接口，可以设置定时任务：

```bash
# 每小时清理一次
0 * * * * curl -X POST http://127.0.0.1:8000/model/cleanup
```

**注意**：由于模型卸载时会自动清理，通常不需要额外的定期清理任务。只有在长时间运行且频繁加载/卸载模型的场景下才建议设置定期清理。

### 清理的进程类型

- `model_worker.py` 相关进程（仅孤立的进程）
- `multiprocessing.spawn` 相关进程（仅LLM模型的遗留进程）
- 占用GPU的Python进程（仅真正的僵尸进程）
- PyTorch/transformers启动的多进程工作进程（仅孤立的进程）

**重要说明**：
- ✅ **Embedding模型保护**：系统会自动识别并保护Embedding模型（如BGE系列）的工作进程
- ✅ **智能区分**：区分LLM模型和Embedding模型的子进程，避免误清理
- ✅ **功能保证**：确保Embedding模型在清理后仍能正常进行推理

---

## 架构优势

- **真正释放显存**：模型卸载后完全释放GPU资源
- **自动端口分配**：由操作系统分配可用端口，避免端口冲突
- **健康检查机制**：确保模型服务正常运行
- **进程崩溃检测**：及时发现并处理进程崩溃
- **异步处理**：使用httpx和asyncio提高并发性能
- **可横向扩展**：支持多机部署
- **可容器化**：易于Docker化部署
- **结构清晰**：模块化设计，易于维护
- **可升级为Kubernetes架构**：支持云原生部署

---

## 注意事项

1. **模型名称格式**：模型名称必须使用 `"组织/模型名"` 的格式，例如 `"Qwen/Qwen3-8B"`
2. **模型路径**：系统会自动根据模型名称从 `MODEL_WEIGHTS_DIR` 拼接完整路径，无需手动指定
3. **端口分配**：每个模型会由操作系统自动分配一个可用端口
4. **自动加载**：如果模型未加载，系统会在调用 `/generate` 或 `/embedding` 接口时自动加载
5. **并发保护**：单个模型的推理请求会串行处理，避免资源冲突
6. **依赖要求**：新增了 `httpx` 和 `psutil` 依赖，用于异步HTTP请求和进程管理
7. **GPU内存管理**：系统已优化进程清理机制，确保模型卸载后完全释放GPU内存
8. **僵尸进程处理**：提供了多种清理僵尸进程的方法，防止GPU内存泄漏
9. **Embedding模型保护**：智能识别并保护Embedding模型的工作进程，确保推理功能正常

---

## Docker部署

本系统提供了完整的Docker部署支持，方便在容器环境中运行和管理。

### 快速开始

1. **构建Docker镜像**

```bash
# 构建Docker镜像
docker build -t llm_deploy:hugface -f docker_build/Dockerfile .
```

2. **运行容器**

```bash
# 赋予执行权限
chmod +x docker_build/start_llm_api.sh

# 运行启动脚本
docker_build/start_llm_api.sh
```

### 详细说明

关于Docker部署的详细配置和管理说明，请参考 `docker_build/README.md` 文件。

---

## 下一步建议

如果您需要进一步扩展系统功能，可以考虑：

- 支持任务队列（防止爆显存）
- 支持GPU使用率监控
- 支持模型自动重启机制
- 支持负载均衡
- 升级成mini私有大模型平台
- 添加认证和授权机制
- 支持流式输出
- 添加日志和监控
- 支持GPU内存实时监控和告警
- 支持模型热加载和版本管理
- 支持分布式模型部署
