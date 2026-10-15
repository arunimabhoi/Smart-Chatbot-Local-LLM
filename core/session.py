"""
core/session.py
---------------
Keeps each browser tab's conversation separate:
    - chat history (memory, like `history` in app_history.py)
    - the FAISS index of the documents that tab uploaded
    - the original documents (needed for summaries)

Everything lives in RAM, so restarting app.py starts fresh.
"""

import threading
import uuid
from dataclasses import dataclass, field

import config


@dataclass
class ChatSession:
    id: str
    history: list = field(default_factory=list)       # [{'role': 'user'|'assistant', 'content': str}]
    vector_store: object = None                       # FAISS index or None
    documents: dict = field(default_factory=dict)     # filename -> [Document, ...]
    chunk_counts: dict = field(default_factory=dict)  # filename -> number of chunks

    def add_turn(self, user_text, assistant_text):
        self.history.append({'role': 'user', 'content': user_text})
        self.history.append({'role': 'assistant', 'content': assistant_text})
        # keep only the most recent messages so the prompt never gets too long
        self.history = self.history[-config.MAX_HISTORY_MESSAGES:]

    def files(self):
        return [{'name': n, 'pages': len(d), 'chunks': self.chunk_counts.get(n, 0)}
                for n, d in self.documents.items()]


_SESSIONS = {}
_LOCK = threading.Lock()


def new_session_id():
    return uuid.uuid4().hex


def get_session(session_id):
    with _LOCK:
        if not session_id:
            session_id = new_session_id()
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = ChatSession(id=session_id)
        return _SESSIONS[session_id]


def reset_session(session_id, keep_documents=True):
    s = get_session(session_id)
    s.history = []
    if not keep_documents:
        s.vector_store = None
        s.documents = {}
        s.chunk_counts = {}
    return s
