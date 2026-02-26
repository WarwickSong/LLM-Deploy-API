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
└── README.md
```

---

## 环境要求

- Python 3.8+
- PyTorch
- FastAPI
- uvicorn
- transformers
- FlagEmbedding
- numpy
- httpx

---

## 安装依赖

```bash
pip install fastapi uvicorn transformers torch flagembedding numpy httpx
```

---

## 配置说明

在启动服务前，请先在 `config.py` 中配置以下参数：

```python
# 主机地址
HOST = "127.0.0.1"

# 服务端口
PORT = 8000

# 健康检查超时时间（秒）
HEALTH_CHECK_TIMEOUT = 120

# 健康检查间隔时间（秒）
HEALTH_CHECK_INTERVAL = 1

# 进程终止超时时间（秒）
PROCESS_TERMINATE_TIMEOUT = 10

# 模型权重根目录
MODEL_WEIGHTS_DIR = r"/data2/home/songzhihua/LLM/llm_playground/model_weights"
```

如果可加载的模型权重有增加，需要在 `models.py` 中的 `MODELS2LOADER` 和 `EMBEDDING2LOADER` 添加对应的模型名称和相应的加载函数。

```python
# 模型名称到加载器的映射
MODELS2LOADER = {  
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

### 模型目录结构

模型权重应按照以下目录结构存放：

```
model_weights/
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

### 2. 加载模型

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

### 3. 文本生成

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

### 4. 文本嵌入

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
```

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
6. **依赖要求**：新增了 `httpx` 依赖，用于异步HTTP请求

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
