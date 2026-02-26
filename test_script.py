#!/usr/bin/env python3
"""
LLM部署系统测试脚本

该脚本用于在远程测试环境中对LLM部署系统进行系统性测试，包括：
1. 环境检查
2. 模型加载/卸载测试
3. 基础功能测试
4. 专用模型类型测试
5. 性能基准测试
6. 测试报告生成

注意：该脚本仅设计为执行测试准备和测试流程，不得在当前环境实际运行。
"""

import argparse
import os
import sys
import json
import time
import requests
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('llm_test')

# 增强的错误处理装饰器
def error_handler(func):
    """错误处理装饰器，捕获并记录函数执行过程中的错误"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"执行 {func.__name__} 时发生错误: {str(e)}")
            logger.debug(f"错误堆栈:", exc_info=True)
            # 修复：增加对results参数的存在性检查
            if 'results' in kwargs and kwargs['results'] is not None:
                results = kwargs['results']
                test_name = f"{func.__name__} 执行错误"
                results.add_result(test_name, False, f"执行 {func.__name__} 时发生错误: {str(e)}", error=str(e))
            # 针对main函数的特殊处理（无results参数）
            elif func.__name__ == 'main':
                logger.error("main函数执行出错，无法记录到TestResults")
            return None
    return wrapper

# 安全的API请求函数
def safe_api_request(method, url, **kwargs):
    """安全的API请求函数，处理各种网络错误"""
    try:
        if method.lower() == 'get':
            response = requests.get(url, **kwargs)
        elif method.lower() == 'post':
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"不支持的HTTP方法: {method}")
        return response
    except requests.exceptions.Timeout:
        logger.error(f"API请求超时: {url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"API连接错误: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求错误: {url}, 错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"未知错误: {url}, 错误: {str(e)}")
        return None

# 全局配置
DEFAULT_API_URL = "http://127.0.0.1:8000"
MODEL_WEIGHTS_DIR = os.environ.get('MODEL_WEIGHTS_DIR', '/data2/home/songzhihua/LLM/llm_playground/model_weights')

# 测试模型列表
TEST_MODELS = {
    "text_models": [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507"
    ],
    "multimodal_models": [
        "Qwen/Qwen3-VL-8B-Thinking"
    ],
    "embedding_models": [
        "BAAI/bge-large-zh-v1.5",
        "BAAI/bge-m3"
    ]
}

# 测试结果存储
class TestResults:
    def __init__(self):
        self.tests = {}
        self.start_time = datetime.now()
        self.end_time = None
        self.summary = {
            "passed": 0,
            "failed": 0,
            "total": 0
        }
        self.test_categories = {
            "environment": [],
            "load_unload": [],
            "basic_functions": [],
            "specific_scenarios": [],
            "performance": []
        }
    
    def add_result(self, test_name: str, status: bool, message: str, 
                   performance_data: Optional[Dict] = None, 
                   error: Optional[str] = None):
        """添加测试结果"""
        self.tests[test_name] = {
            "status": "PASSED" if status else "FAILED",
            "message": message,
            "performance_data": performance_data,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.summary["total"] += 1
        if status:
            self.summary["passed"] += 1
        else:
            self.summary["failed"] += 1
        
        # 分类测试结果
        if "环境检查" in test_name:
            self.test_categories["environment"].append(test_name)
        elif "模型加载" in test_name or "模型卸载" in test_name:
            self.test_categories["load_unload"].append(test_name)
        elif "基础功能" in test_name or "文本生成" in test_name or "问答交互" in test_name:
            self.test_categories["basic_functions"].append(test_name)
        elif "专用测试" in test_name:
            self.test_categories["specific_scenarios"].append(test_name)
        elif "性能基准" in test_name:
            self.test_categories["performance"].append(test_name)
    
    def finalize(self):
        """完成测试，记录结束时间"""
        self.end_time = datetime.now()
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.end_time - self.start_time) if self.end_time else None,
            "summary": self.summary,
            "test_categories": self.test_categories,
            "tests": self.tests
        }
    
    def generate_html_report(self, output_file: str):
        """生成HTML格式的测试报告"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM部署系统测试报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .summary {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary-item {
            display: inline-block;
            margin-right: 30px;
            font-size: 16px;
        }
        .summary-item .value {
            font-weight: bold;
            font-size: 20px;
        }
        .passed {
            color: green;
        }
        .failed {
            color: red;
        }
        .category {
            margin-bottom: 30px;
        }
        .category h2 {
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
        .test-item {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 4px solid #ddd;
        }
        .test-item.passed {
            border-left-color: green;
        }
        .test-item.failed {
            border-left-color: red;
        }
        .test-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .test-message {
            color: #666;
            margin-bottom: 5px;
        }
        .test-error {
            color: red;
            font-style: italic;
            margin-top: 5px;
        }
        .performance-data {
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 14px;
        }
        .timestamp {
            color: #999;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM部署系统测试报告</h1>
        
        <div class="summary">
            <div class="summary-item">
                开始时间: <span class="value">{{start_time}}</span>
            </div>
            <div class="summary-item">
                结束时间: <span class="value">{{end_time}}</span>
            </div>
            <div class="summary-item">
                测试时长: <span class="value">{{duration}}</span>
            </div>
            <div class="summary-item">
                总测试数: <span class="value">{{total}}</span>
            </div>
            <div class="summary-item">
                通过: <span class="value passed">{{passed}}</span>
            </div>
            <div class="summary-item">
                失败: <span class="value failed">{{failed}}</span>
            </div>
            <div class="summary-item">
                成功率: <span class="value">{{success_rate}}%</span>
            </div>
        </div>
        
        {{categories}}
    </div>
</body>
</html>
        """
        
        # 计算成功率
        success_rate = round((self.summary["passed"] / self.summary["total"]) * 100, 2) if self.summary["total"] > 0 else 0
        
        # 生成分类测试结果
        categories_html = ""
        for category_name, test_names in self.test_categories.items():
            if not test_names:
                continue
            
            category_title = {
                "environment": "环境检查",
                "load_unload": "模型加载/卸载",
                "basic_functions": "基础功能",
                "specific_scenarios": "专用场景",
                "performance": "性能基准"
            }.get(category_name, category_name)
            
            tests_html = ""
            for test_name in test_names:
                test = self.tests.get(test_name)
                if not test:
                    continue
                
                # 修复：添加续行符，修正字符串拼接语法错误
                performance_html = ""
                if test.get("performance_data"):
                    performance_html = "<div class='performance-data'>" + \
                                      json.dumps(test["performance_data"], indent=2, ensure_ascii=False) + \
                                      "</div>"
                
                error_html = ""
                if test.get("error"):
                    error_html = f"<div class='test-error'>错误: {test['error']}</div>"
                
                # 修复：移除空的f-string拼接
                tests_html += f"<div class='test-item {test['status'].lower()}'>"
                tests_html += f"<div class='test-name'>{test_name}</div>"
                tests_html += f"<div class='test-message'>{test['message']}</div>"
                tests_html += performance_html
                tests_html += error_html
                tests_html += f"<div class='timestamp'>{test['timestamp']}</div>"
                tests_html += "</div>"
            
            # 修复：移除空的f-string拼接
            categories_html += f"<div class='category'>"
            categories_html += f"<h2>{category_title}</h2>"
            categories_html += tests_html
            categories_html += "</div>"
        
        # 填充模板
        html_content = html_template.replace("{{start_time}}", self.start_time.isoformat())
        html_content = html_content.replace("{{end_time}}", self.end_time.isoformat() if self.end_time else "N/A")
        html_content = html_content.replace("{{duration}}", str(self.end_time - self.start_time) if self.end_time else "N/A")
        html_content = html_content.replace("{{total}}", str(self.summary["total"]))
        html_content = html_content.replace("{{passed}}", str(self.summary["passed"]))
        html_content = html_content.replace("{{failed}}", str(self.summary["failed"]))
        html_content = html_content.replace("{{success_rate}}", str(success_rate))
        html_content = html_content.replace("{{categories}}", categories_html)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def generate_detailed_report(self, output_file: str):
        """生成详细的JSON测试报告"""
        report_data = self.to_dict()
        
        # 添加更多统计信息
        report_data["statistics"] = {
            "success_rate": round((self.summary["passed"] / self.summary["total"]) * 100, 2) if self.summary["total"] > 0 else 0,
            "category_statistics": {}
        }
        
        # 计算每个分类的统计信息
        for category_name, test_names in self.test_categories.items():
            if not test_names:
                continue
            
            category_passed = 0
            category_failed = 0
            
            for test_name in test_names:
                test = self.tests.get(test_name)
                if test and test["status"] == "PASSED":
                    category_passed += 1
                elif test:
                    category_failed += 1
            
            report_data["statistics"]["category_statistics"][category_name] = {
                "total": len(test_names),
                "passed": category_passed,
                "failed": category_failed,
                "success_rate": round((category_passed / len(test_names)) * 100, 2) if len(test_names) > 0 else 0
            }
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return output_file

# 测试脚本配置
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM部署系统测试脚本')
    
    # 基本配置
    parser.add_argument('--api-url', type=str, default=DEFAULT_API_URL,
                      help='API服务地址')
    parser.add_argument('--model-dir', type=str, default=MODEL_WEIGHTS_DIR,
                      help='模型权重目录')
    parser.add_argument('--output', type=str, default='test_report/test_report.json',
                      help='测试报告输出文件')
    
    # 测试范围
    parser.add_argument('--test-env', action='store_true', default=True,
                      help='执行环境检查测试')
    parser.add_argument('--test-load', action='store_true', default=True,
                      help='执行模型加载/卸载测试')
    parser.add_argument('--test-basic', action='store_true', default=True,
                      help='执行基础功能测试')
    parser.add_argument('--test-specific', action='store_true', default=True,
                      help='执行模型类型专用测试')
    parser.add_argument('--test-performance', action='store_true', default=True,
                      help='执行性能基准测试')
    
    # 测试深度
    parser.add_argument('--depth', type=int, default=1,
                      help='测试深度 (1-3，数字越大测试越全面)')
    
    # 模型选择
    parser.add_argument('--models', type=str, nargs='+',
                      help='指定要测试的模型列表')
    
    return parser.parse_args()

# 环境检查模块
def check_environment(api_url: str, model_dir: str, depth: int, results: TestResults):
    """检查测试环境的依赖项、硬件资源和模型文件完整性"""
    
    # 1. 检查Python依赖项
    def check_dependencies():
        """检查必要的Python依赖项"""
        required_packages = [
            'requests', 'psutil', 'torch', 'transformers', 'FlagEmbedding'
        ]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            message = f"缺少依赖项: {', '.join(missing_packages)}"
            results.add_result("依赖项检查", False, message)
            return False
        else:
            results.add_result("依赖项检查", True, "所有必要的依赖项已安装")
            return True
    
    # 2. 检查硬件资源
    def check_hardware():
        """检查硬件资源"""
        # 检查内存
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        
        # 检查CPU
        cpu_count = psutil.cpu_count(logical=True)
        
        # 检查GPU (尝试导入torch)
        gpu_available = False
        gpu_count = 0
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
        except ImportError:
            pass
        
        hardware_info = {
            "memory_gb": round(memory_gb, 2),
            "available_memory_gb": round(memory.available / (1024 ** 3), 2),
            "cpu_count": cpu_count,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count
        }
        
        # 根据测试深度确定硬件要求
        if depth >= 2:
            if memory_gb < 16:
                results.add_result("硬件资源检查", False, "内存不足，建议至少16GB", hardware_info)
                return False
            if not gpu_available:
                results.add_result("硬件资源检查", False, "未检测到GPU，大型模型可能无法加载", hardware_info)
                return False
        
        results.add_result("硬件资源检查", True, "硬件资源检查通过", hardware_info)
        return True
    
    # 3. 检查模型文件完整性
    def check_model_files():
        """检查模型文件是否完整"""
        model_paths = {
            "Qwen/Qwen3-8B": os.path.join(model_dir, "Qwen", "Qwen3-8B"),
            "Qwen/Qwen3-30B-A3B-Instruct-2507": os.path.join(model_dir, "Qwen", "Qwen3-30B-A3B-Instruct-2507"),
            "Qwen/Qwen3-VL-8B-Thinking": os.path.join(model_dir, "Qwen", "Qwen3-VL-8B-Thinking"),
            "BAAI/bge-large-zh-v1.5": os.path.join(model_dir, "BAAI", "bge-large-zh-v1.5"),
            "BAAI/bge-m3": os.path.join(model_dir, "BAAI", "bge-m3")
        }
        
        missing_models = []
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                missing_models.append(model_name)
        
        if missing_models:
            message = f"缺少模型文件: {', '.join(missing_models)}"
            results.add_result("模型文件检查", False, message)
            return False
        else:
            results.add_result("模型文件检查", True, "所有模型文件路径存在")
            return True
    
    # 4. 检查API服务可访问性
    def check_api_access():
        """检查API服务是否可访问"""
        # 修复：使用safe_api_request替代直接requests调用
        response = safe_api_request('get', f"{api_url}/model/list", timeout=10)
        if response and response.status_code == 200:
            results.add_result("API服务检查", True, "API服务可正常访问")
            return True
        else:
            message = f"无法访问API服务: 响应为空或状态码错误"
            results.add_result("API服务检查", False, message)
            return False
    
    # 执行所有环境检查
    check_dependencies()
    check_hardware()
    check_model_files()
    check_api_access()

# 模型加载/卸载测试模块
def test_model_load_unload(api_url: str, models: List[str], depth: int, results: TestResults):
    """测试模型加载/卸载功能，确保每个模型都能正确加载/卸载，卸载时释放相关资源"""
    
    def get_available_models():
        """获取所有可用的模型列表"""
        response = safe_api_request('get', f"{api_url}/model/list", timeout=10)
        if response and response.status_code == 200:
            try:
                return list(response.json().keys())
            except Exception:
                return []
        else:
            return []
    
    def load_model(model_name: str):
        """加载指定模型"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        response = safe_api_request(
            'post',
            f"{api_url}/model/load",
            json={"model_name": model_name},
            timeout=300  # 给大型模型足够的加载时间
        )
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        if response and response.status_code == 200:
            performance_data = {
                "load_time_seconds": round(end_time - start_time, 2),
                "memory_used_mb": round((end_memory - start_memory) / (1024 ** 2), 2)
            }
            return True, performance_data
        else:
            error_msg = response.text if response else "请求失败"
            return False, {"error": error_msg}
    
    def unload_model(model_name: str):
        """卸载指定模型"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        response = safe_api_request(
            'post',
            f"{api_url}/model/unload",
            json={"model_name": model_name},
            timeout=60
        )
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        if response and response.status_code == 200:
            performance_data = {
                "unload_time_seconds": round(end_time - start_time, 2),
                "memory_freed_mb": round((start_memory - end_memory) / (1024 ** 2), 2)
            }
            return True, performance_data
        else:
            error_msg = response.text if response else "请求失败"
            return False, {"error": error_msg}
    
    # 确定要测试的模型列表
    test_models = models if models else []
    if not test_models:
        # 如果没有指定模型，测试所有支持的模型
        test_models = []
        test_models.extend(TEST_MODELS["text_models"])
        test_models.extend(TEST_MODELS["multimodal_models"])
        test_models.extend(TEST_MODELS["embedding_models"])
    
    # 测试每个模型的加载/卸载
    for model_name in test_models:
        logger.info(f"测试模型: {model_name}")
        
        # 1. 先确保模型未加载
        available_models = get_available_models()
        if model_name in available_models:
            success, perf_data = unload_model(model_name)
            if not success:
                logger.warning(f"卸载已存在的模型 {model_name} 失败: {perf_data.get('error')}")
        
        # 2. 测试模型加载
        logger.info(f"加载模型: {model_name}")
        success, perf_data = load_model(model_name)
        
        if success:
            results.add_result(
                f"模型加载测试 - {model_name}", 
                True, 
                f"模型 {model_name} 加载成功",
                perf_data
            )
            
            # 3. 验证模型是否成功加载
            available_models = get_available_models()
            if model_name in available_models:
                results.add_result(
                    f"模型加载验证 - {model_name}", 
                    True, 
                    f"模型 {model_name} 在可用模型列表中"
                )
            else:
                results.add_result(
                    f"模型加载验证 - {model_name}", 
                    False, 
                    f"模型 {model_name} 不在可用模型列表中"
                )
            
            # 4. 测试模型卸载
            logger.info(f"卸载模型: {model_name}")
            success, perf_data = unload_model(model_name)
            
            if success:
                results.add_result(
                    f"模型卸载测试 - {model_name}", 
                    True, 
                    f"模型 {model_name} 卸载成功",
                    perf_data
                )
                
                # 5. 验证模型是否成功卸载
                available_models = get_available_models()
                if model_name not in available_models:
                    results.add_result(
                        f"模型卸载验证 - {model_name}", 
                        True, 
                        f"模型 {model_name} 已从可用模型列表中移除"
                    )
                else:
                    results.add_result(
                        f"模型卸载验证 - {model_name}", 
                        False, 
                        f"模型 {model_name} 仍在可用模型列表中"
                    )
            else:
                results.add_result(
                    f"模型卸载测试 - {model_name}", 
                    False, 
                    f"模型 {model_name} 卸载失败: {perf_data.get('error')}"
                )
        else:
            results.add_result(
                f"模型加载测试 - {model_name}", 
                False, 
                f"模型 {model_name} 加载失败: {perf_data.get('error')}"
            )

        # 在深度测试模式下，等待一段时间让资源完全释放
        if depth >= 2:
            time.sleep(5)

# 基础功能测试模块
def test_basic_functions(api_url: str, depth: int, results: TestResults):
    """测试基础功能，包括文本生成、问答交互等核心功能验证"""
    
    def test_text_generation(model_name: str):
        """测试文本生成功能"""
        test_cases = [
            {
                "name": "简单问候",
                "messages": [{"role": "user", "content": "你好，很高兴认识你！"}],
                "max_new_tokens": 512
            },
            {
                "name": "技术问题",
                "messages": [{"role": "user", "content": "什么是人工智能？"}],
                "max_new_tokens": 1024
            },
            {
                "name": "创意写作",
                "messages": [{"role": "user", "content": "写一个关于未来科技的短篇故事，开头是：'在2150年的一个早晨，李明被窗外的声音吵醒了...'"}],
                "max_new_tokens": 1536
            }
        ]
        
        # 根据测试深度调整测试用例数量
        if depth < 2:
            test_cases = test_cases[:2]
        
        for test_case in test_cases:
            test_name = f"文本生成测试 - {model_name} - {test_case['name']}"
            logger.info(f"执行测试: {test_name}")
            
            start_time = time.time()
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/generate",
                    json={
                        "model_name": model_name,
                        "messages": test_case["messages"],
                        "max_new_tokens": test_case["max_new_tokens"]
                    },
                    timeout=300
                )
                
                end_time = time.time()
                
                if response and response.status_code == 200:
                    result_data = response.json()
                    if "response" in result_data and result_data["response"]:
                        performance_data = {
                            "response_time_seconds": round(end_time - start_time, 2),
                            "response_length": len(str(result_data["response"]))
                        }
                        results.add_result(
                            test_name, 
                            True, 
                            f"文本生成测试通过，模型返回了有效响应",
                            performance_data
                        )
                    else:
                        results.add_result(
                            test_name, 
                            False, 
                            f"文本生成测试失败，响应格式不正确: {result_data}"
                        )
                else:
                    results.add_result(
                        test_name, 
                        False, 
                        f"文本生成测试失败，状态码: {response.status_code if response else '无响应'}, 响应: {response.text if response else '无'}"
                    )
            except Exception as e:
                results.add_result(
                    test_name, 
                    False, 
                    f"文本生成测试失败，异常: {str(e)}"
                )
    
    def test_qa_interaction(model_name: str):
        """测试问答交互功能"""
        test_cases = [
            {
                "name": "常识问题",
                "messages": [
                    {"role": "user", "content": "中国的首都是哪里？"}
                ],
                "expected_keywords": ["北京"]
            },
            {
                "name": "多轮对话",
                "messages": [
                    {"role": "user", "content": "什么是Python？"},
                    {"role": "assistant", "content": "Python是一种高级编程语言。"},
                    {"role": "user", "content": "它有什么特点？"}
                ],
                "expected_keywords": ["简单", "易学", "可读性", "丰富的库"]
            }
        ]
        
        for test_case in test_cases:
            test_name = f"问答交互测试 - {model_name} - {test_case['name']}"
            logger.info(f"执行测试: {test_name}")
            
            start_time = time.time()
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/generate",
                    json={
                        "model_name": model_name,
                        "messages": test_case["messages"],
                        "max_new_tokens": 1024
                    },
                    timeout=300
                )
                
                end_time = time.time()
                
                if response and response.status_code == 200:
                    result_data = response.json()
                    if "response" in result_data and result_data["response"]:
                        response_text = str(result_data["response"]).lower()
                        
                        # 检查是否包含期望的关键词
                        if "expected_keywords" in test_case:
                            found_keywords = [
                                keyword for keyword in test_case["expected_keywords"] 
                                if keyword.lower() in response_text
                            ]
                            
                            if found_keywords:
                                performance_data = {
                                    "response_time_seconds": round(end_time - start_time, 2),
                                    "found_keywords": found_keywords,
                                    "total_keywords": len(test_case["expected_keywords"])
                                }
                                results.add_result(
                                    test_name, 
                                    True, 
                                    f"问答交互测试通过，找到 {len(found_keywords)} 个期望关键词",
                                    performance_data
                                )
                            else:
                                results.add_result(
                                    test_name, 
                                    False, 
                                    f"问答交互测试失败，未找到期望的关键词",
                                    {"response": result_data["response"]}
                                )
                        else:
                            performance_data = {
                                "response_time_seconds": round(end_time - start_time, 2)
                            }
                            results.add_result(
                                test_name, 
                                True, 
                                f"问答交互测试通过，模型返回了有效响应",
                                performance_data
                            )
                    else:
                        results.add_result(
                            test_name, 
                            False, 
                            f"问答交互测试失败，响应格式不正确: {result_data}"
                        )
                else:
                    results.add_result(
                        test_name, 
                        False, 
                        f"问答交互测试失败，状态码: {response.status_code if response else '无响应'}, 响应: {response.text if response else '无'}"
                    )
            except Exception as e:
                results.add_result(
                    test_name, 
                    False, 
                    f"问答交互测试失败，异常: {str(e)}"
                )
    
    # 测试的模型列表
    test_models = ["Qwen/Qwen3-8B"]  # 选择一个基础文本模型进行测试
    
    for model_name in test_models:
        logger.info(f"测试模型: {model_name}")
        
        # 确保模型已加载
        try:
            # 修复：使用safe_api_request替代直接requests调用
            response = safe_api_request(
                'post',
                f"{api_url}/model/load",
                json={"model_name": model_name},
                timeout=300
            )
            if not response or response.status_code != 200:
                raise Exception("加载请求失败")
        except Exception:
            logger.warning(f"加载模型 {model_name} 失败，跳过基础功能测试")
            results.add_result(
                f"基础功能测试 - {model_name}", 
                False, 
                f"模型 {model_name} 加载失败，无法进行基础功能测试"
            )
            continue
        
        # 执行测试
        test_text_generation(model_name)
        test_qa_interaction(model_name)
        
        # 卸载模型
        try:
            # 修复：使用safe_api_request替代直接requests调用
            response = safe_api_request(
                'post',
                f"{api_url}/model/unload",
                json={"model_name": model_name},
                timeout=60
            )
        except Exception:
            logger.warning(f"卸载模型 {model_name} 失败")

# 模型类型专用测试模块
def test_model_specific_scenarios(api_url: str, depth: int, results: TestResults):
    """针对不同模型类型设计专用测试场景"""
    
    def test_text_models():
        """测试文本模型的专用场景"""
        text_models = TEST_MODELS["text_models"]
        
        for model_name in text_models:
            logger.info(f"测试文本模型: {model_name}")
            
            # 确保模型已加载
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/load",
                    json={"model_name": model_name},
                    timeout=300
                )
                if not response or response.status_code != 200:
                    logger.warning(f"加载模型 {model_name} 失败")
                    continue
            except Exception:
                logger.warning(f"加载模型 {model_name} 失败")
                continue
            
            # 测试长文本生成
            test_name = f"文本模型专用测试 - {model_name} - 长文本生成"
            logger.info(f"执行测试: {test_name}")
            
            start_time = time.time()
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/generate",
                    json={
                        "model_name": model_name,
                        "messages": [{"role": "user", "content": "写一篇关于人工智能发展历史的文章，从1950年代开始，到2020年代结束，详细介绍每个时期的重要突破和代表人物。"}],
                        "max_new_tokens": 2048
                    },
                    timeout=600
                )
                
                end_time = time.time()
                
                if response and response.status_code == 200:
                    result_data = response.json()
                    if "response" in result_data and result_data["response"]:
                        response_text = str(result_data["response"])
                        performance_data = {
                            "response_time_seconds": round(end_time - start_time, 2),
                            "response_length": len(response_text)
                        }
                        
                        # 检查响应长度
                        if len(response_text) > 1000:
                            results.add_result(
                                test_name, 
                                True, 
                                f"长文本生成测试通过，生成了 {len(response_text)} 个字符",
                                performance_data
                            )
                        else:
                            results.add_result(
                                test_name, 
                                False, 
                                f"长文本生成测试失败，生成的文本过短: {len(response_text)} 个字符",
                                performance_data
                            )
                    else:
                        results.add_result(
                            test_name, 
                            False, 
                            f"长文本生成测试失败，响应格式不正确: {result_data}"
                        )
                else:
                    results.add_result(
                        test_name, 
                        False, 
                        f"长文本生成测试失败，状态码: {response.status_code if response else '无响应'}, 响应: {response.text if response else '无'}"
                    )
            except Exception as e:
                results.add_result(
                    test_name, 
                    False, 
                    f"长文本生成测试失败，异常: {str(e)}"
                )
            
            # 卸载模型
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/unload",
                    json={"model_name": model_name},
                    timeout=60
                )
            except Exception:
                logger.warning(f"卸载模型 {model_name} 失败")
    
    def test_multimodal_models():
        """测试多模态模型的专用场景"""
        multimodal_models = TEST_MODELS["multimodal_models"]
        
        for model_name in multimodal_models:
            logger.info(f"测试多模态模型: {model_name}")
            
            # 确保模型已加载
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/load",
                    json={"model_name": model_name},
                    timeout=300
                )
                if not response or response.status_code != 200:
                    logger.warning(f"加载模型 {model_name} 失败")
                    continue
            except Exception:
                logger.warning(f"加载模型 {model_name} 失败")
                continue
            
            # 测试思考模式（Thinking Mode）
            test_name = f"多模态模型专用测试 - {model_name} - 思考模式"
            logger.info(f"执行测试: {test_name}")
            
            start_time = time.time()
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/generate",
                    json={
                        "model_name": model_name,
                        "messages": [{"role": "user", "content": "如果x=5，y=3，那么x²+y³等于多少？请详细思考过程。"}],
                        "max_new_tokens": 1024
                    },
                    timeout=300
                )
                
                end_time = time.time()
                
                if response and response.status_code == 200:
                    result_data = response.json()
                    if "thinking" in result_data and "response" in result_data:
                        performance_data = {
                            "response_time_seconds": round(end_time - start_time, 2),
                            "has_thinking": bool(result_data["thinking"]),
                            "has_response": bool(result_data["response"])
                        }
                        
                        if result_data["thinking"]:
                            results.add_result(
                                test_name, 
                                True, 
                                f"思考模式测试通过，模型生成了思考过程",
                                performance_data
                            )
                        else:
                            results.add_result(
                                test_name, 
                                False, 
                                f"思考模式测试失败，模型没有生成思考过程",
                                performance_data
                            )
                    else:
                        results.add_result(
                            test_name, 
                            False, 
                            f"思考模式测试失败，响应格式不正确: {result_data}"
                        )
                else:
                    results.add_result(
                        test_name, 
                        False, 
                        f"思考模式测试失败，状态码: {response.status_code if response else '无响应'}, 响应: {response.text if response else '无'}"
                    )
            except Exception as e:
                results.add_result(
                    test_name, 
                    False, 
                    f"思考模式测试失败，异常: {str(e)}"
                )
            
            # 卸载模型
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/unload",
                    json={"model_name": model_name},
                    timeout=60
                )
            except Exception:
                logger.warning(f"卸载模型 {model_name} 失败")
    
    def test_embedding_models():
        """测试嵌入模型的专用场景"""
        embedding_models = TEST_MODELS["embedding_models"]
        
        for model_name in embedding_models:
            logger.info(f"测试嵌入模型: {model_name}")
            
            # 确保模型已加载
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/load",
                    json={"model_name": model_name},
                    timeout=300
                )
                if not response or response.status_code != 200:
                    logger.warning(f"加载模型 {model_name} 失败")
                    continue
            except Exception:
                logger.warning(f"加载模型 {model_name} 失败")
                continue
            
            # 测试文本嵌入
            test_name = f"嵌入模型专用测试 - {model_name} - 文本嵌入"
            logger.info(f"执行测试: {test_name}")
            
            start_time = time.time()
            try:
                test_texts = ["这是一个测试句子", "这是另一个测试句子", "这是第三个测试句子"]
                
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/embedding",
                    json={
                        "model_name": model_name,
                        "texts": test_texts,
                        "query": False
                    },
                    timeout=300
                )
                
                end_time = time.time()
                
                if response and response.status_code == 200:
                    result_data = response.json()
                    if "embedding" in result_data:
                        embeddings = result_data["embedding"]
                        performance_data = {
                            "response_time_seconds": round(end_time - start_time, 2),
                            "num_embeddings": len(embeddings),
                            "embedding_dim": len(embeddings[0]) if embeddings else 0
                        }
                        
                        # 验证嵌入结果
                        if len(embeddings) == len(test_texts) and all(isinstance(emb, list) for emb in embeddings):
                            results.add_result(
                                test_name, 
                                True, 
                                f"文本嵌入测试通过，生成了 {len(embeddings)} 个嵌入向量",
                                performance_data
                            )
                        else:
                            results.add_result(
                                test_name, 
                                False, 
                                f"文本嵌入测试失败，嵌入结果格式不正确",
                                performance_data
                            )
                    else:
                        results.add_result(
                            test_name, 
                            False, 
                            f"文本嵌入测试失败，响应格式不正确: {result_data}"
                        )
                else:
                    results.add_result(
                        test_name, 
                        False, 
                        f"文本嵌入测试失败，状态码: {response.status_code if response else '无响应'}, 响应: {response.text if response else '无'}"
                    )
            except Exception as e:
                results.add_result(
                    test_name, 
                    False, 
                    f"文本嵌入测试失败，异常: {str(e)}"
                )
            
            # 卸载模型
            try:
                # 修复：使用safe_api_request替代直接requests调用
                response = safe_api_request(
                    'post',
                    f"{api_url}/model/unload",
                    json={"model_name": model_name},
                    timeout=60
                )
            except Exception:
                logger.warning(f"卸载模型 {model_name} 失败")
    
    # 执行各类模型的专用测试
    test_text_models()
    test_multimodal_models()
    test_embedding_models()

# 性能基准测试模块
def test_performance_benchmark(api_url: str, depth: int, results: TestResults):
    """测试性能基准，记录模型响应时间、内存占用等关键指标"""
    
    def measure_performance(model_name: str, test_type: str):
        """测量模型的性能指标"""
        # 确保模型已加载
        try:
            # 修复：使用safe_api_request替代直接requests调用
            response = safe_api_request(
                'post',
                f"{api_url}/model/load",
                json={"model_name": model_name},
                timeout=300
            )
            if not response or response.status_code != 200:
                logger.warning(f"加载模型 {model_name} 失败")
                return None
        except Exception:
            logger.warning(f"加载模型 {model_name} 失败")
            return None
        
        # 等待模型完全加载
        time.sleep(5)
        
        measurements = []
        iterations = 3 if depth >= 2 else 2
        
        for i in range(iterations):
            logger.info(f"执行性能测试迭代 {i+1}/{iterations} - {model_name} - {test_type}")
            
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            
            try:
                if test_type == "text_generation":
                    # 测试文本生成性能
                    # 修复：使用safe_api_request替代直接requests调用
                    response = safe_api_request(
                        'post',
                        f"{api_url}/generate",
                        json={
                            "model_name": model_name,
                            "messages": [{"role": "user", "content": "解释什么是机器学习，以及它的主要应用领域。"}],
                            "max_new_tokens": 512
                        },
                        timeout=300
                    )
                elif test_type == "embedding":
                    # 测试嵌入性能
                    # 修复：使用safe_api_request替代直接requests调用
                    response = safe_api_request(
                        'post',
                        f"{api_url}/embedding",
                        json={
                            "model_name": model_name,
                            "texts": ["这是一个测试句子", "这是另一个测试句子", "这是第三个测试句子"],
                            "query": False
                        },
                        timeout=300
                    )
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                
                if response and response.status_code == 200:
                    measurement = {
                        "iteration": i+1,
                        "response_time_seconds": round(end_time - start_time, 2),
                        "memory_used_mb": round((end_memory - start_memory) / (1024 ** 2), 2),
                        "success": True
                    }
                    measurements.append(measurement)
                else:
                    logger.warning(f"性能测试迭代失败: {response.text if response else '无响应'}")
            except Exception as e:
                logger.warning(f"性能测试迭代异常: {str(e)}")
            
            # 等待资源释放
            time.sleep(3)
        
        # 卸载模型
        try:
            # 修复：使用safe_api_request替代直接requests调用
            response = safe_api_request(
                'post',
                f"{api_url}/model/unload",
                json={"model_name": model_name},
                timeout=60
            )
        except Exception:
            logger.warning(f"卸载模型 {model_name} 失败")
        
        # 计算平均性能指标
        if measurements:
            avg_response_time = sum(m["response_time_seconds"] for m in measurements) / len(measurements)
            avg_memory_used = sum(m["memory_used_mb"] for m in measurements) / len(measurements)
            
            return {
                "measurements": measurements,
                "average_response_time_seconds": round(avg_response_time, 2),
                "average_memory_used_mb": round(avg_memory_used, 2),
                "total_iterations": len(measurements),
                "successful_iterations": len([m for m in measurements if m["success"]])
            }
        else:
            return None
    
    # 测试文本模型性能
    text_models = TEST_MODELS["text_models"]
    for model_name in text_models:
        logger.info(f"测试文本模型性能: {model_name}")
        performance_data = measure_performance(model_name, "text_generation")
        
        if performance_data:
            results.add_result(
                f"性能基准测试 - {model_name} - 文本生成",
                True,
                f"性能测试完成，平均响应时间: {performance_data['average_response_time_seconds']}秒",
                performance_data
            )
        else:
            results.add_result(
                f"性能基准测试 - {model_name} - 文本生成",
                False,
                f"性能测试失败，无法获取性能数据"
            )
    
    # 测试嵌入模型性能
    embedding_models = TEST_MODELS["embedding_models"]
    for model_name in embedding_models:
        logger.info(f"测试嵌入模型性能: {model_name}")
        performance_data = measure_performance(model_name, "embedding")
        
        if performance_data:
            results.add_result(
                f"性能基准测试 - {model_name} - 文本嵌入",
                True,
                f"性能测试完成，平均响应时间: {performance_data['average_response_time_seconds']}秒",
                performance_data
            )
        else:
            results.add_result(
                f"性能基准测试 - {model_name} - 文本嵌入",
                False,
                f"性能测试失败，无法获取性能数据"
            )

# 主函数（仅保留一个完整版本）
@error_handler
def main():
    """测试脚本主函数"""
    args = parse_args()
    results = TestResults()
    
    try:
        logger.info(f"开始测试LLM部署系统，API地址: {args.api_url}")
        logger.info(f"测试深度: {args.depth}")
        
        # 验证API服务是否可访问
        api_response = safe_api_request('get', f"{args.api_url}/model/list", timeout=10)
        if not api_response:
            logger.error("无法连接到API服务，测试将终止")
            results.add_result("API服务连接测试", False, "无法连接到API服务")
            return
        
        # 1. 环境检查
        if args.test_env:
            logger.info("执行环境检查测试")
            check_environment(args.api_url, args.model_dir, args.depth, results)
        
        # 2. 模型加载/卸载测试
        if args.test_load:
            logger.info("执行模型加载/卸载测试")
            test_models = args.models if args.models else []
            test_model_load_unload(args.api_url, test_models, args.depth, results)
        
        # 3. 基础功能测试
        if args.test_basic:
            logger.info("执行基础功能测试")
            test_basic_functions(args.api_url, args.depth, results)
        
        # 4. 模型类型专用测试
        if args.test_specific:
            logger.info("执行模型类型专用测试")
            test_model_specific_scenarios(args.api_url, args.depth, results)
        
        # 5. 性能基准测试
        if args.test_performance:
            logger.info("执行性能基准测试")
            test_performance_benchmark(args.api_url, args.depth, results)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        results.add_result("测试中断", False, "测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行过程中发生错误: {str(e)}")
        results.add_result("测试执行错误", False, f"测试执行过程中发生错误: {str(e)}", error=str(e))
    finally:
        # 生成测试报告
        results.finalize()
        
        # 生成详细的JSON报告
        detailed_output = args.output.replace('.json', '_detailed.json')
        results.generate_detailed_report(detailed_output)
        
        # 生成HTML报告
        html_output = args.output.replace('.json', '.html')
        results.generate_html_report(html_output)
        
        # 生成简洁的JSON报告
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试完成，报告已生成:")
        logger.info(f"- 简洁JSON报告: {args.output}")
        logger.info(f"- 详细JSON报告: {detailed_output}")
        logger.info(f"- HTML报告: {html_output}")
        logger.info(f"测试结果: 总测试数={results.summary['total']}, 通过={results.summary['passed']}, 失败={results.summary['failed']}")
        
        # 计算并显示成功率
        if results.summary['total'] > 0:
            success_rate = round((results.summary['passed'] / results.summary['total']) * 100, 2)
            logger.info(f"测试成功率: {success_rate}%")
        
        # 显示失败的测试
        failed_tests = [name for name, test in results.tests.items() if test['status'] == 'FAILED']
        if failed_tests:
            logger.warning(f"失败的测试: {', '.join(failed_tests[:5])}")
            if len(failed_tests) > 5:
                logger.warning(f"... 还有 {len(failed_tests) - 5} 个失败的测试")

if __name__ == "__main__":
    main()