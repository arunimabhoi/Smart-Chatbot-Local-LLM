"""
core/llm.py
-----------
Creates the local Ollama models.               (Langchain_Fundamentals.ipynb)

    llm        -> factual answers  (temperature 0)  used by RAG, summary
    chat_llm   -> friendly chat    (temperature 0.7)
    embeddings -> turns text into vectors for FAISS  (Simple_RAG.ipynb)
"""

import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config


def get_llm(temperature=None, model=None):
    """Build a ChatOllama model. Nothing is downloaded or paid for - it talks to your local Ollama."""
    kwargs = dict(
        model=model or config.CHAT_MODEL,
        temperature=config.FACTUAL_TEMPERATURE if temperature is None else temperature,
        base_url=config.OLLAMA_URL,
        num_ctx=config.NUM_CTX,
    )
    if config.THINKING is not None:
        kwargs['reasoning'] = config.THINKING      # False -> no hidden "thinking", no empty answers
    return ChatOllama(**kwargs)


def get_embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


# Shared instances (creating them does NOT contact Ollama yet)
llm        = get_llm(temperature=config.FACTUAL_TEMPERATURE)
chat_llm   = get_llm(temperature=config.CHAT_TEMPERATURE)
agent_llm  = get_llm(temperature=0, model=config.AGENT_MODEL)
embeddings = get_embeddings()


def check_ollama():
    """
    Asks Ollama which models are installed.
    Returns a dict the web page uses to show a green / red status light.
    """
    needed = {config.CHAT_MODEL, config.AGENT_MODEL, config.EMBED_MODEL}
    try:
        resp = requests.get(config.OLLAMA_URL + '/api/tags', timeout=3)
        resp.raise_for_status()
        installed = [m['name'] for m in resp.json().get('models', [])]
    except Exception as e:
        return {'running': False, 'installed': [], 'missing': sorted(needed),
                'error': 'Ollama is not reachable at {} ({})'.format(config.OLLAMA_URL, e)}

    def is_installed(name):
        # 'nomic-embed-text' and 'nomic-embed-text:latest' are the same model
        base = name if ':' in name else name + ':latest'
        return name in installed or base in installed

    missing = sorted(m for m in needed if not is_installed(m))
    return {'running': True, 'installed': installed, 'missing': missing, 'error': None}
