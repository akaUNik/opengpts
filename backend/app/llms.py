import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
import httpx
import structlog
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.chat_models.ollama import ChatOllama

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_gigachat_llm():
    authorization = os.environ.get("GIGACHAT_API_KEY")
    auth_url = os.environ.get("GIGACHAT_AUTH_URL")
    scope = os.environ.get("GIGACHAT_SCOPE")
    model = os.environ.get("GIGACHAT_MODEL")

    return GigaChat(
        credentials=authorization,
        auth_url=auth_url,
        scope=scope,
        temperature=0.01,
        verbose=True,
        profanity_check=False,
        model=model,
        timeout=3600
    )


@lru_cache(maxsize=1)
def get_ollama_llm():
    model_name = os.environ.get("OLLAMA_MODEL")
    if not model_name:
        model_name = "llama2"
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    if not ollama_base_url:
        ollama_base_url = "http://localhost:11434"

    return ChatOllama(model=model_name, base_url=ollama_base_url)
