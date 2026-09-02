"""Embedding 模型的懒加载封装，统一本地模型和 API 模型入口。

本地 HuggingFace 模型会固定到配置指定的设备；``auto`` 优先选择 CUDA，
显式指定 CUDA 但运行环境不支持时直接失败，避免索引任务误用 CPU。
"""
import threading

from app.config import settings

# 进程级单例：本地模型加载昂贵（读盘 + HF hub 网络检查），
# 不能每次请求都新建，否则高延迟且在外网不可达时会触发重试风暴。
_instance = None
_lock = threading.Lock()


def _resolve_local_device() -> str:
    """解析本地模型设备，并在显式 CUDA 不可用时给出明确错误。"""
    import torch

    configured = settings.embedding_device
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "EMBEDDING_DEVICE 已要求使用 CUDA，但当前 PyTorch 不支持 CUDA；"
            "请安装 CUDA 版 torch，或暂时设置 EMBEDDING_DEVICE=cpu。"
        )
    return configured


def get_embedding_device() -> str:
    """返回本地 Embedding 实际使用的设备；API Embedding 返回 ``api``。"""
    if settings.embedding_provider != "local":
        return "api"
    return _resolve_local_device()


def get_embeddings():
    """返回 LangChain Embeddings 实例（OpenAI 兼容接口或本地 bge，进程级单例）。"""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        if settings.embedding_provider == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            device = _resolve_local_device()
            model_name = settings.embedding_local_path or settings.embedding_model
            _instance = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "batch_size": settings.embedding_batch_size,
                    "normalize_embeddings": True,
                },
                # langchain-huggingface 会单独传 show_progress_bar，不能重复放入 encode_kwargs。
                show_progress=False,
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            _instance = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
                timeout=settings.embedding_timeout,  # 单次请求超时（秒）
                max_retries=settings.embedding_max_retries,  # 失败自动重试
                # 第三方 OpenAI-compatible 服务未必使用 OpenAI tokenizer；关闭后把原始字符串
                # 交给服务端 tokenizer，避免 tiktoken token ID 被误解释。
                check_embedding_ctx_length=settings.embedding_check_ctx_length,
            )
    return _instance
