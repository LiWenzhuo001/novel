"""统一创建聊天模型实例，并集中处理模型、超时和 API 配置。"""
from app.config import settings
from langchain_openai import ChatOpenAI


def get_llm(
    streaming: bool = False,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """返回 LangChain ChatOpenAI（兼容 DeepSeek / 通义 / 智谱等 OpenAI 接口）。

    ``model`` 用于同 base_url 下的模型覆盖（如 LLM-as-judge 的 ``settings.judge_model``）；
    为 None 时使用 settings.llm_model。
    """
    settings.validate()
    kwargs = {
        "model": model or settings.llm_model,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": temperature,
        "streaming": streaming,
        "timeout": settings.llm_timeout if timeout is None else timeout,
        "max_retries": settings.llm_max_retries if max_retries is None else max_retries,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # DeepSeek V4 等推理模型关闭思考模式（thinking.type=disabled）：
    # 直接返回 content，避免思维链空 content + 长延迟。对不支持该参数的服务端会被忽略，不报错。
    if settings.llm_disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)
