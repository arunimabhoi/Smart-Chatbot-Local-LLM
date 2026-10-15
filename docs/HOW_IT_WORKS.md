# How the notebooks became this project

Your instructor's plan was to take each notebook and turn it into one module.
Here is the map, so you can open a notebook and the matching file side by side.

| Notebook / script | Concept | Where it lives now |
|---|---|---|
| `Langchain_Fundamentals.ipynb` | `ChatOllama`, `invoke`, System/Human messages | `core/llm.py`, `core/agent.py → stream_chat` |
| `Langchain_Fundamentals.ipynb` | `PromptTemplate`, `prompt \| llm \| parser` chains | every `*_chain` in `core/` |
| `Langchain_Fundamentals.ipynb` | `chain.stream()` | all answers stream to the browser token by token |
| `Langchain_Fundamentals.ipynb` | Email writer with name / purpose / tone | `core/prompt_lab.py → email_chain` |
| `Langchain_Fundamentals.ipynb` | "Explain {topic} to a {person}" | `core/prompt_lab.py → explain_chain` |
| `Prompting_techniques.ipynb` | Zero-shot classification | `core/prompt_lab.py → sentiment_chain` |
| `Prompting_techniques.ipynb` | Few-shot (`FewShotChatMessagePromptTemplate`) | `core/prompt_lab.py → emotion_chain` |
| `Prompting_techniques.ipynb` | Structured output format (fields) | `core/summarizer.py → SUMMARY_PROMPT` |
| `Prompting_techniques.ipynb` | Chained prompts (summarise → translate) | `core/summarizer.py` map → reduce, `translate_chain` |
| `Prompting_techniques.ipynb` | Chain-of-thought | chat system prompt ("reason step by step"), calculator tool |
| `Simple_RAG.ipynb` | `Document` + metadata | `core/loader.py` |
| `Simple_RAG.ipynb` | `RecursiveCharacterTextSplitter` | `core/vector_store.py → text_splitter` |
| `Simple_RAG.ipynb` | `OllamaEmbeddings` + `FAISS` | `core/vector_store.py → build_or_extend` |
| `Simple_RAG.ipynb` | RAG prompt + `format_docs` | `core/vector_store.py → RAG_PROMPT, format_docs` |
| `AI_Agent.ipynb` | `@tool` get_datetime, wikipedia, DuckDuckGo | `core/tools.py` (+ new `calculator`) |
| `AI_Agent.ipynb` | `create_agent(model, tools, system_prompt)` | `core/agent.py → agent` |
| `AI_Agent.ipynb` | `run_agent(show_steps=True)` | `core/agent.py → stream_agent` ("Agent steps" in the UI) |
| `app_history.py` | Chat history → System/Human/AI messages | `core/session.py` + `core/agent.py → to_messages` |
| `app_simple.py` / `app_history.py` | Gradio chat UI | replaced by Flask + `templates/index.html` (supports file upload, modes, steps) |

## Changes worth knowing (good interview talking points)

1. **`reasoning=False` on ChatOllama.** Several notebook cells printed an empty line. qwen3/qwen3.5 are
   "thinking" models; the answer went into hidden reasoning. Turning thinking off fixes it.
2. **Map-reduce summarisation.** A 2B model cannot read a 50-page PDF at once. Long files are summarised
   part by part, then the part-summaries are summarised (`core/summarizer.py`).
3. **Sources in RAG.** Each answer shows which file and page the chunks came from, which makes the answers checkable.
4. **Calculator tool.** Small models make arithmetic mistakes (see the train-seats cell). The agent now calls Python for maths.
5. **Sessions.** Each browser tab has its own chat history and its own FAISS index.
6. **Streaming over HTTP.** Flask sends one JSON object per line (NDJSON); `script.js` reads them with
   `fetch().body.getReader()` and paints tokens as they arrive.

## Ideas to extend it

- Save the FAISS index to disk with `vector_store.save_local()` so documents survive a restart.
- Add a `search_documents` tool so the Web Agent can also look inside your uploads.
- Use `ChatPromptTemplate` with `MessagesPlaceholder` to give RAG answers full chat history.
- Add a model picker drop-down that reads `ollama list` via `/api/health`.
