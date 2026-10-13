import os
from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_llm(temperature: float = 0) -> BaseChatModel:
    """
    Returns the configured chat LLM. Defaults to OpenAI (proven reliable for
    this app's forced-tool-calling agents). Set LLM_PROVIDER=groq to use Groq
    instead - verified live that the only model a standard Groq key can
    reach (openai/gpt-oss-120b; Llama requires an enterprise account) both
    rejects this app's tool-calling requests over an 8,000 TPM cap on even a
    tiny corpus, and 400s on the Supervisor's forced structured-output call.
    Safe to flip once you have real Llama 3.3 70B access or a model that
    handles forced tool-calling reliably - see BENCHMARK_RESULTS.md.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq
        # Llama 3.3 70B is Groq's flagship model, but as of this writing it
        # (and llama-3.1-8b-instant) require an enterprise-tier account -
        # a standard API key gets a 404. openai/gpt-oss-120b is Groq-hosted
        # and works on standard keys; override via GROQ_MODEL once your
        # account has Llama access.
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'groq' or 'openai')")
