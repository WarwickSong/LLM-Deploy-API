# 基础镜像仍用miniconda（保留conda环境能力，仅用pip装依赖）
FROM continuumio/miniconda3:latest

# 1. 保持原有工作目录和PYTHONPATH配置（完全不变）
RUN mkdir -p /app/workdir
WORKDIR /app/workdir
ENV PYTHONPATH="${PYTHONPATH}:/app/workdir"

# 2. 创建空的conda环境（仅保留环境名hugface，不用yml文件）
# 指定python版本（替换成你本地环境的版本，比如3.10）
RUN conda create -n hugface python=3.10.9 -y

# 3. 复制pip依赖清单（替换原来的environment.yml）
# 注意：你需要先在本地执行 pip freeze > requirements.txt 生成这个文件
COPY requirements.txt /app/requirements.txt

# 4. 激活conda环境并安装pip依赖（关键修改）
# 使用conda run在hugface环境中执行pip安装
RUN conda run -n hugface pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt

# 5. 保持原有入口命令（完全不变）
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hugface", "python", "main_api.py"]