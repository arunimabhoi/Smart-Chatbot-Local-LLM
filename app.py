"""
app.py
------
The web server. Run:   python app.py   then open  http://127.0.0.1:5000

Routes
    GET  /                 -> the chat page (templates/index.html)
    GET  /api/health       -> is Ollama running? are the models installed?
    POST /api/upload       -> upload documents, index them in FAISS
    POST /api/summarize    -> stream a summary of one uploaded file
    POST /api/chat         -> stream an answer (chat / documents / agent / smart actions)
    POST /api/reset        -> clear the conversation (and optionally the documents)

Answers are streamed as "NDJSON": one small JSON object per line, e.g.
    {"type": "token", "text": "Hel"}
    {"type": "token", "text": "lo!"}
    {"type": "done"}
"""

import json
import os
import traceback

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

import config
from core.llm import check_ollama
from core.loader import load_file
from core.vector_store import build_or_extend, stream_rag_answer
from core.summarizer import stream_summary
from core.agent import stream_chat, stream_agent
from core.prompt_lab import stream_action
from core.session import get_session, reset_session

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

SMART_ACTIONS = {'sentiment', 'translate', 'email', 'explain'}


def ndjson_stream(events, on_finish=None):
    """
    Turns a generator of event dicts into a streaming HTTP response.
    Collects the answer text so it can be saved into the chat history at the end.
    """
    def generate():
        answer = []
        try:
            for event in events:
                if event.get('type') == 'token':
                    answer.append(event['text'])
                yield json.dumps(event) + '\n'
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({'type': 'error', 'text': friendly_error(e)}) + '\n'
        full = ''.join(answer).strip()
        if on_finish and full:
            on_finish(full)
        yield json.dumps({'type': 'done'}) + '\n'

    return Response(stream_with_context(generate()),
                    mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def friendly_error(e):
    msg = str(e)
    low = msg.lower()
    if 'connect' in low or 'refused' in low or 'connection' in low:
        return 'Cannot reach Ollama. Open a terminal and run:  ollama serve'
    if 'not found' in low and ('model' in low or 'pull' in low):
        return 'Model not installed. {}  (run: ollama pull <model-name>)'.format(msg)
    if 'does not support tools' in low:
        return ('This model cannot call tools. Set AGENT_MODEL in config.py to a '
                'tool-capable model such as qwen3.5:2b or llama3.2.')
    if 'thinking' in low:
        return 'This model does not support the thinking switch. Set THINKING = None in config.py.'
    return 'Something went wrong: {}'.format(msg)


# ------------------------------------------------------------------ pages
@app.get('/')
def index():
    return render_template('index.html',
                           chat_model=config.CHAT_MODEL,
                           embed_model=config.EMBED_MODEL,
                           allowed=', '.join('.' + e for e in sorted(config.ALLOWED_EXTENSIONS)),
                           max_mb=config.MAX_UPLOAD_MB)


@app.get('/api/health')
def health():
    status = check_ollama()
    status.update({'chat_model': config.CHAT_MODEL,
                   'agent_model': config.AGENT_MODEL,
                   'embed_model': config.EMBED_MODEL})
    return jsonify(status)


# ------------------------------------------------------------------ upload
@app.post('/api/upload')
def upload():
    session = get_session(request.form.get('session_id'))
    files = request.files.getlist('files')
    if not files:
        return jsonify({'ok': False, 'error': 'No file received.'}), 400

    folder = os.path.join(config.UPLOAD_DIR, session.id)
    os.makedirs(folder, exist_ok=True)

    added, errors = [], []
    for f in files:
        name = secure_filename(f.filename or '')
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if ext not in config.ALLOWED_EXTENSIONS:
            errors.append('{}: unsupported file type'.format(f.filename))
            continue
        path = os.path.join(folder, name)
        f.save(path)
        try:
            docs = load_file(path, name)
            session.vector_store, n_chunks = build_or_extend(session.vector_store, docs)
            session.documents[name] = docs
            session.chunk_counts[name] = n_chunks
            added.append(name)
        except Exception as e:
            traceback.print_exc()
            errors.append('{}: {}'.format(name, friendly_error(e) if 'connect' in str(e).lower() else e))

    return jsonify({'ok': bool(added), 'added': added, 'errors': errors,
                    'files': session.files(), 'session_id': session.id})


# ------------------------------------------------------------------ summarize
@app.post('/api/summarize')
def summarize():
    data = request.get_json(force=True)
    session = get_session(data.get('session_id'))
    filename = data.get('filename')
    docs = session.documents.get(filename)
    if not docs:
        return jsonify({'error': 'File not found - please upload it again.'}), 404

    request_text = 'Summarize {}'.format(filename)
    return ndjson_stream(stream_summary(docs, filename),
                         on_finish=lambda ans: session.add_turn(request_text, ans))


# ------------------------------------------------------------------ chat
@app.post('/api/chat')
def chat():
    data = request.get_json(force=True)
    session = get_session(data.get('session_id'))
    message = (data.get('message') or '').strip()
    mode = data.get('mode', 'chat')
    option = data.get('option', '')
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    history = list(session.history)

    if mode == 'docs':
        events = stream_rag_answer(session.vector_store, message, history)
    elif mode == 'agent':
        events = stream_agent(message, history)
    elif mode in SMART_ACTIONS:
        events = stream_action(mode, message, option)
    else:
        events = stream_chat(message, history)

    return ndjson_stream(events, on_finish=lambda ans: session.add_turn(message, ans))


@app.post('/api/reset')
def reset():
    data = request.get_json(force=True)
    s = reset_session(data.get('session_id'), keep_documents=data.get('keep_documents', True))
    return jsonify({'ok': True, 'files': s.files()})


if __name__ == '__main__':
    status = check_ollama()
    print('=' * 60)
    print(' Smart Chatbot')
    print('   Chat model  :', config.CHAT_MODEL)
    print('   Embeddings  :', config.EMBED_MODEL)
    if not status['running']:
        print('   WARNING     : Ollama is not running -> start it with:  ollama serve')
    elif status['missing']:
        for m in status['missing']:
            print('   MISSING     : {}   ->  ollama pull {}'.format(m, m))
    else:
        print('   Ollama      : OK')
    print('   Open        : http://{}:{}'.format(config.HOST, config.PORT))
    print('=' * 60)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
