"""
config.py
---------
ONE place for every setting in the project.
Change the model names here to match what you have in `ollama list`.
No API keys are needed anywhere - everything runs on your own computer.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Ollama models  (run `ollama list` to see what you have installed)
# ------------------------------------------------------------------
OLLAMA_URL   = os.getenv('OLLAMA_URL', 'http://localhost:11434')

CHAT_MODEL   = os.getenv('CHAT_MODEL', 'qwen3.5:2b')            # used for chat, RAG, summary
AGENT_MODEL  = os.getenv('AGENT_MODEL', CHAT_MODEL)              # must support tool-calling (qwen does)
EMBED_MODEL  = os.getenv('EMBED_MODEL', 'nomic-embed-text:latest')

CHAT_TEMPERATURE    = 0.7    # creative answers in normal chat
FACTUAL_TEMPERATURE = 0.0    # RAG, summary, agent -> stick to the facts

# qwen3 / qwen3.5 are "thinking" models. In your notebooks some answers came
# back EMPTY - that was the model spending its output on hidden thinking.
# False = switch thinking off (fast, visible answers).
# If a model ever complains that it "does not support thinking", set this to None.
THINKING = False

# How many tokens the model can "see" at once. Bigger = longer documents,
# but more RAM. 8192 is a good middle ground for 2B-4B models.
NUM_CTX = 8192

# ------------------------------------------------------------------
# RAG settings (from Simple_RAG.ipynb)
# ------------------------------------------------------------------
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 4            # how many chunks to retrieve per question

# ------------------------------------------------------------------
# Summarizer
# ------------------------------------------------------------------
# Documents longer than this (in characters) are summarised in two passes:
# first each part, then the summary of the parts (map -> reduce).
SUMMARY_DIRECT_LIMIT = 12000
SUMMARY_PART_SIZE    = 6000

# ------------------------------------------------------------------
# Chat memory
# ------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 12    # last 6 questions + 6 answers are remembered

# ------------------------------------------------------------------
# Uploads
# ------------------------------------------------------------------
UPLOAD_DIR         = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'docx', 'csv'}
MAX_UPLOAD_MB      = 25

# ------------------------------------------------------------------
# Web server
# ------------------------------------------------------------------
HOST  = '127.0.0.1'
PORT  = 5000
DEBUG = False
