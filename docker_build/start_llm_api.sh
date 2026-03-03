#!/bin/bash
# 停止并删除旧容器（避免端口占用）
docker stop llm_api_server > /dev/null 2>&1
docker rm llm_api_server > /dev/null 2>&1

# 启动 llm_api_server 容器（后台运行）
# --name：指定容器名称，便于管理
# --gpus：启用所有 GPU
# --shm-size：共享内存大小
# -p：宿主机 8000 端口映射到容器 8000 端口（容器内服务监听 8000）
# -v：挂载宿主机上的项目脚本目录到容器内的 /app/workdir
# -v：挂载宿主机上的模型权重目录到容器内的 /app/model_weights
# 使用镜像 llm_deploy:hugface

docker run -d \
  --name llm_api_server \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v WarwickSong/llm_deploy_api:/app/workdir \
  -v WarwickSong/model_weights:/app/model_weights \
  llm_deploy:hugface

# 验证启动结果
echo "容器启动中，3秒后查看状态..."
sleep 3
docker ps | grep llm_api_server && echo "✅ 服务启动成功，访问地址：http://服务器IP:8000"

# # NOTE: 保存上述内容为start_llm_api.sh
# chmod +x start_llm_api.sh  # 赋予执行权限
# ./start_llm_api.sh         # 一键启动

# # 常用管理命令（配套使用）
# # 1. 查看服务实时日志（调试必备）
# docker logs -f llm_api_server

# # 2. 停止服务
# docker stop llm_api_server

# # 3. 重启服务（修改代码后无需重建镜像，重启即可）
# docker restart llm_api_server

# # 4. 进入容器调试（比如检查依赖/修改临时配置）
# docker exec -it llm_api_server bash

# # 5. 彻底删除容器（停止后清理）
# docker rm llm_api_server