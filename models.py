"""
模型加载和推理模块

该模块负责LLM模型的加载、推理和嵌入生成，包括：
- 支持多种Qwen系列模型（Qwen3-8B、Qwen3-30B-A3B-Instruct-2507、Qwen3-VL系列等）
- 支持BGE系列嵌入模型（bge-large-zh-v1.5、bge-m3）
- 提供文本生成和文本嵌入功能
- 支持思考模式（Thinking Mode）
"""

import numpy as np
import torch
from FlagEmbedding import FlagModel  # bge-large-zh-v1.5; bge-m3 
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3ForCausalLM,  # Qwen3-8B
    Qwen3MoeForCausalLM,  # Qwen3-30B-A3B-Instruct-2507
    Qwen3VLForConditionalGeneration,   # Qwen3-VL-32B-Instruct
    Qwen3VLMoeForConditionalGeneration,   # Qwen3-VL-30B-A3B-Instruct
)

# 应对pytorch版本不匹配Qwen的修复：强制绑定is_compiling属性（关键）
torch.compiler.is_compiling = lambda: False  # 固定返回False，不影响推理

# =========================================================
# 配置区
# =========================================================
ENABLE_THINKING = False  # 关闭思考模式，不设置则默认开启
THINKING_TOKEN_ID = 151668  # </think>标记的token ID
MAX_NEW_TOKENS = 8192  # 最大新生成token数
MIN_PIXELS = 256*28*28 
MAX_PIXELS = 1536*28*28
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

# =========================
# 加载模型和处理器
# =========================
def load_model_and_processor(model_name: str, model_path: str):
    """
    加载指定模型和处理器

    Args:
        model_name: 模型名称，格式为"组织/模型名"
        model_path: 模型权重路径

    Returns:
        tuple: (model, processor) 模型实例和处理器实例

    Raises:
        ValueError: 当模型名称不支持时抛出异常
    """
    model_loader = MODELS2LOADER.get(model_name)
    if not model_loader:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 加载模型
    model = model_loader.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    # 确保模型在评估模式
    model.eval()
    
    return model, processor


def load_embedding_model(model_name: str, model_path: str):
    """
    加载指定的embedding模型

    Args:
        model_name: 模型名称，格式为"组织/模型名"
        model_path: 模型权重路径

    Returns:
        FlagModel: 加载的embedding模型实例

    Raises:
        ValueError: 当模型名称不支持时抛出异常
    """
    embedding_loader = EMBEDDING2LOADER.get(model_name)
    if not embedding_loader:
        raise ValueError(f"不支持的embedding模型: {model_name}")
    
    # 加载embedding模型
    model = embedding_loader.from_pretrained(
        model_path,
        query_instruction_for_retrieval="为这个文本生成表示以用于检索相关内容：",
        use_fp16=True,
        trust_remote_code=True,  # 必须加，适配自定义模型
        use_safetensors=True     # 强制使用safetensors，避开torch版本限制
    )
    
    # Embedding模型采用懒加载，在进行编码时才真正加载
    (
        model.encode_queries('query_1'), 
        model.encode('query_1'),
        model.encode_queries(['query_1', 'query_2']), 
        model.encode(['query_1', 'query_2'])
    )
    return model


# =========================================================
# 模型推理
# =========================================================
def generate_response(
    model, 
    processor, 
    messages: list, 
    enable_thinking: bool = ENABLE_THINKING, 
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[list[str], list[str]]:
    """
    生成模型响应，支持思考模式

    Args:
        model: 加载的模型实例
        processor: 加载的处理器实例
        messages: 输入消息列表，每个消息是一个字典，包含"role"和"content"。其中"role"是"user"或"assistant", "content"是消息内容。如果模型为VL模型，则"content"是一个列表，包含图片和文本；如果模型为非VL模型，则"content"必须是一个字符串。
        enable_thinking: 是否启用思考模式，默认值为ENABLE_THINKING
        max_new_tokens: 最大新生成token数，默认值为MAX_NEW_TOKENS

    Returns:
        tuple: (thinking, response)
            - thinking: 模型生成的思考字符串列表
            - response: 模型生成的响应字符串列表
    """
    # 准备模型输入
    model_inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking  # 启用思考模式
    )
    model_inputs = model_inputs.to(model.device)
    
    # 模型推理
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
    # 提取生成的token ID
    generated_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    # 解析思考内容和回答内容
    index = []
    for out_ids in generated_ids:
        try:
            # 查找</think>标记的位置
            index.append(len(out_ids) - out_ids.tolist()[::-1].index(THINKING_TOKEN_ID))
        except ValueError:
            index.append(0)
    
    # 解析思考内容和回答内容
    thinking = [
        processor.decode(out_ids[:i], skip_special_tokens=True).strip()  
        for out_ids, i in zip(generated_ids, index)
    ]
    # 解析回答内容
    response = [
        processor.decode(out_ids[i:], skip_special_tokens=True).strip() 
        for out_ids, i in zip(generated_ids, index)
    ]

    return thinking, response


def get_embedding(
    model, 
    queries: list[str], 
    batch_size: int = 32,
    query: bool = False,
) -> np.ndarray:
    """
    获取模型对查询文本的embedding表示

    Args:
        model: 加载的embedding模型实例
        queries: 输入查询文本列表，每个元素是一个字符串
        batch_size: 批次大小，默认值为32
        query: 是否为查询文本，默认值为False：- 如果为True，则使用encode_queries方法编码查询文本；- 如果为False，则使用encode方法编码普通文本。

    Returns:
        np.ndarray: 模型对查询文本的embedding表示，形状为(n_queries, embedding_dim)
    """
    
    # 分批处理查询文本
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        with torch.no_grad():
            # 根据是否为查询文本选择不同的编码方法
            if query:
                batch_embeddings = model.encode_queries(batch_queries)
            else:
                batch_embeddings = model.encode(batch_queries)
        embeddings.append(batch_embeddings)
    
    # 合并所有批次的embedding
    embeddings = np.concatenate(embeddings, axis=0)
    
    return embeddings
