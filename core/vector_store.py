"""
core/vector_store.py
--------------------
RAG = "chat with your documents".                       (Simple_RAG.ipynb)

    split into chunks  ->  embed with nomic-embed-text  ->  store in FAISS
    question  ->  find the closest chunks  ->  give them to the LLM as context
"""

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from core.llm import llm, embeddings


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,        # max characters per chunk
    chunk_overlap=config.CHUNK_OVERLAP,  # overlap so sentences are not cut in half
    length_function=len,
)


def build_or_extend(vector_store, documents):
    """
    Splits the documents and adds them to the FAISS index.
    First upload creates the index; later uploads are added to the same one,
    so you can ask questions across several files.
    Returns (vector_store, number_of_new_chunks).
    """
    chunks = text_splitter.split_documents(documents)
    if vector_store is None:
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    else:
        vector_store.add_documents(chunks)
    return vector_store, len(chunks)


# ---------------- RAG prompt (same idea as the notebook) ----------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    'You are a helpful AI assistant. Answer the user\'s question using ONLY the context below.\n'
    'If the answer is not in the context, say: "I don\'t have enough information in the '
    'uploaded documents to answer this."\n'
    'Mention the file name (and page) you used, like (report.pdf, p.3).\n\n'
    'CONVERSATION SO FAR:\n{history}\n\n'
    'CONTEXT:\n{context}\n\n'
    'QUESTION: {question}\n'
    'ANSWER:'
)

rag_chain = RAG_PROMPT | llm | StrOutputParser()


def format_docs(docs):
    return '\n\n'.join(
        'Source : {} (page {})\n{}'.format(d.metadata.get('source'), d.metadata.get('page'), d.page_content)
        for d in docs
    )


def format_history(history):
    if not history:
        return '(none)'
    return '\n'.join('{}: {}'.format(m['role'].upper(), m['content'][:500]) for m in history[-6:])


def stream_rag_answer(vector_store, question, history=None):
    """
    Yields events for the browser:
        {'type': 'sources', 'sources': [...]}   - which chunks were used
        {'type': 'token',   'text': '...'}      - the answer, piece by piece
    """
    if vector_store is None:
        yield {'type': 'token',
               'text': 'Please upload a document first (left panel), then ask your question.'}
        return

    docs = vector_store.similarity_search(question, k=config.TOP_K)

    yield {'type': 'sources', 'sources': [
        {'source': d.metadata.get('source'), 'page': d.metadata.get('page'),
         'preview': d.page_content[:220]}
        for d in docs
    ]}

    for text in rag_chain.stream({
        'context': format_docs(docs),
        'question': question,
        'history': format_history(history),
    }):
        if text:
            yield {'type': 'token', 'text': text}
