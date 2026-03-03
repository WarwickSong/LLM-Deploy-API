# Docker部署指南

本目录包含LLM部署系统的Docker构建和运行配置文件，用于快速部署和管理LLM模型服务。

## 目录结构

```
docker_build/
├── Dockerfile          # Docker镜像构建文件
├── requirements.txt    # Python依赖包列表
├── start_llm_api.sh    # 启动脚本
└── README.md           # 本说明文件
```

## 环境要求

- Docker 19.03+  
- NVIDIA Docker运行时（用于GPU支持）
- 足够的磁盘空间（用于存储Docker镜像和模型权重）

## 构建Docker镜像

### 1. 准备工作

1. 确保已安装Docker和NVIDIA Docker运行时
2. 准备好模型权重目录（参考主README.md的模型目录结构）
3. 确保当前目录在 `llm_deploy_api` 目录下

### 2. 构建镜像

```bash
# 构建Docker镜像
docker build -t llm_deploy:hugface -f docker_build/Dockerfile .

# 查看构建的镜像
docker images | grep llm_deploy
```

## 配置说明

### Dockerfile

Dockerfile使用以下配置：
- 基础镜像：`continuumio/miniconda3:latest`
- Python版本：3.10.9
- 工作目录：`/app/workdir`
- 模型权重目录：`/app/model_weights`（在容器内）
- 依赖安装：使用清华源加速安装

### requirements.txt

包含以下主要依赖：
- fastapi==0.134.0
- FlagEmbedding==1.3.5
- httpx==0.28.1
- numpy==1.23.5
- psutil==5.9.0
- pydantic==2.12.5
- Requests==2.32.5
- torch==2.2.1
- transformers==4.57.3
- uvicorn==0.41.0

### start_llm_api.sh

启动脚本包含以下功能：
- 停止并删除旧容器
- 启动新容器（后台运行）
- 映射端口：宿主机8000 → 容器8000
- 挂载目录：
  - 项目脚本目录：`WarwickSong/llm_deploy_api` → `/app/workdir`
  - 模型权重目录：`WarwickSong/model_weights` → `/app/model_weights`
- 验证启动结果

## 运行服务

### 1. 修改启动脚本（可选）

根据实际情况修改 `start_llm_api.sh` 中的挂载路径：

```bash
# 修改为你的实际路径
-v /path/to/your/llm_deploy_api:/app/workdir 
-v /path/to/your/model_weights:/app/model_weights 
```

### 2. 赋予执行权限并运行

```bash
# 赋予执行权限
chmod +x docker_build/start_llm_api.sh

# 运行启动脚本
docker_build/start_llm_api.sh
```

### 3. 验证服务

服务启动后，可通过以下方式验证：

```bash
# 查看容器状态
docker ps | grep llm_api_server

# 查看服务日志
docker logs -f llm_api_server

# 测试API
curl http://localhost:8000/model/list
```

## 常用管理命令

### 1. 查看服务实时日志

```bash
docker logs -f llm_api_server
```

### 2. 停止服务

```bash
docker stop llm_api_server
```

### 3. 重启服务

```bash
docker restart llm_api_server
```

### 4. 进入容器调试

```bash
docker exec -it llm_api_server bash
```

### 5. 彻底删除容器

```bash
docker rm llm_api_server
```

## 注意事项

1. **GPU支持**：确保主机已安装NVIDIA驱动和NVIDIA Docker运行时
2. **模型权重**：确保模型权重目录结构正确，符合主README.md的要求
3. **端口冲突**：如果8000端口已被占用，需修改启动脚本中的端口映射
4. **资源需求**：根据加载的模型大小，确保主机有足够的GPU内存
5. **网络访问**：如果需要从外部访问服务，确保防火墙已开放8000端口

## 故障排除

### 1. 容器启动失败

- 检查Docker日志：`docker logs llm_api_server`
- 确保模型权重路径正确
- 确保宿主机和容器内都是GPU可用：`nvidia-smi`

### 2. 模型加载失败

- 检查模型权重是否完整
- 检查GPU内存是否足够
- 查看容器日志获取详细错误信息

### 3. API调用失败

- 检查服务是否正常运行：`docker ps | grep llm_api_server`
- 检查网络连接：`curl http://localhost:8000/health`
- 查看服务日志获取详细错误信息

## 升级说明

当代码或依赖发生变化时，需要重新构建镜像：

```bash
# 停止并删除旧容器
docker stop llm_api_server
docker rm llm_api_server

# 重新构建镜像
docker build -t llm_deploy:hugface -f docker_build/Dockerfile .

# 启动新容器
docker_build/start_llm_api.sh
```