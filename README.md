# Smart Chatbot (100% local, Ollama + LangChain)

A ChatGPT-style web app that runs completely on your own computer.
**No API keys, no payments** — every model comes from Ollama.

What it can do:

| Mode | What it does | Built from |
|---|---|---|
| **Chat** | Normal conversation that remembers earlier messages | `Langchain_Fundamentals.ipynb`, `app_history.py` |
| **Ask documents** | Upload PDF / Word / TXT / MD / CSV and ask questions about them (RAG). Shows which file & page was used | `Simple_RAG.ipynb` |
| **Summarize** (button next to each file) | Streams a structured summary: Overview, Key Points, Conclusion. Long files are summarised part by part first | `Langchain_Fundamentals.ipynb` (streaming), `Prompting_techniques.ipynb` |
| **Web Agent** | Decides by itself whether to use web search (DuckDuckGo), Wikipedia, the clock or a calculator, and shows each step | `AI_Agent.ipynb` |
| **Email / Translate / Explain / Sentiment** | Quick "smart actions" | `Prompting_techniques.ipynb`, `Langchain_Fundamentals.ipynb` |

---

## 1. One-time setup

### a) Ollama models
Ollama must be installed and running. In a terminal:

```bash
ollama pull qwen3.5:2b          # chat model (you already used this in app_history.py)
ollama pull nomic-embed-text    # embedding model for documents (used in Simple_RAG)
ollama list                     # check they are there
```

Want different models? Change `CHAT_MODEL` / `EMBED_MODEL` at the top of **`config.py`**.
Good choices for a normal laptop: `qwen3.5:2b`, `qwen3.5:4b`, `llama3.2:3b`.
`gemma3:1b` works for Chat but **cannot use tools**, so keep a qwen or llama model for the Web Agent (`AGENT_MODEL`).

### b) Python environment (Windows)

Open a terminal **inside the `Smart_Chatbot` folder**:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(On Mac/Linux use `source venv/bin/activate`.)

You can also reuse your existing course environment (`D:\ARU\TUTORT ACADEMY\GenAI\Langchain`) — it already has most packages. Then just run `pip install flask pypdf python-docx`.

## 2. Run it

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.
The dot in the top-left turns green when Ollama and both models are found. If it is red, it tells you exactly which `ollama pull ...` command to run.

## 3. Try it

1. **Chat:** "Explain overfitting with an example" → then "give me a shorter version" (it remembers).
2. **Documents:** drag a PDF into the left panel → click **Summarize** → then ask "What are the main numbers in this report?"
3. **Web Agent:** "What is today's date?" · "Current gold price in India?" · "Who was Napoleon?" — open *Agent steps* under the answer to see which tool it used.
4. **Email:** pick the *Email* pill, choose a tone, type "Ask my manager for 2 days leave".
5. A sample file to test with is in `docs/sample_ai_notes.txt`.

---

## Project structure

```
Smart_Chatbot/
├── app.py               Flask web server + streaming API
├── config.py            ALL settings: model names, chunk size, etc.
├── requirements.txt
├── core/
│   ├── llm.py           ChatOllama + OllamaEmbeddings, Ollama health check
│   ├── loader.py        PDF / DOCX / TXT / MD / CSV  ->  LangChain Documents
│   ├── vector_store.py  split -> embed -> FAISS, RAG prompt + streaming answer
│   ├── summarizer.py    streaming summary (map -> reduce for long files)
│   ├── tools.py         get_datetime, web_search, wikipedia, calculator
│   ├── agent.py         chat with memory + tool-calling agent (create_agent)
│   ├── prompt_lab.py    zero-shot, few-shot, chained prompts (smart actions)
│   └── session.py       per-tab chat history and document index
├── templates/index.html chat page
├── static/script.js     reads the streamed answer and shows it live
├── static/style.css
├── uploads/             your uploaded files are saved here
└── docs/                how it works + a sample document
```

## How a message flows

```
Browser (script.js) ──POST /api/chat──▶ app.py ──▶ chosen mode
                                                    ├─ chat   → agent.stream_chat   → chat_llm.stream()
                                                    ├─ docs   → vector_store.stream_rag_answer → FAISS search → llm.stream()
                                                    ├─ agent  → agent.stream_agent  → create_agent(tools)
                                                    └─ action → prompt_lab.stream_action
         ◀── one JSON line per piece of text ("token") ──┘
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Red dot: "Ollama not running" | Start the Ollama app, or run `ollama serve` |
| "Missing: ollama pull ..." | Run that exact command |
| Answers are empty | Keep `THINKING = False` in `config.py` (qwen "thinking" hides the answer) |
| "does not support thinking" error | Set `THINKING = None` in `config.py` |
| Agent says it "cannot call tools" | Set `AGENT_MODEL = 'qwen3.5:2b'` (or `llama3.2`) in `config.py` |
| Upload says "No text found" | It is a scanned PDF (images). Use a PDF with selectable text |
| Slow answers | Normal for CPU. Try a smaller model, or reduce `NUM_CTX` to 4096 |
| Web search fails | Needs internet; DuckDuckGo sometimes rate-limits — wait a minute |

Everything is kept in memory: restarting `app.py` clears chats and indexes (uploaded files stay in `uploads/`).
