"""
core/summarizer.py
------------------
Summarises an uploaded document and STREAMS it to the browser.
(Chain + streaming from Langchain_Fundamentals.ipynb, structured prompt from
Prompting_techniques.ipynb.)

Small local models can only read so much at once, so:
    short document -> summarise it in one go
    long document  -> MAP: summarise each part  ->  REDUCE: summarise the summaries
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from core.llm import llm


SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    'You are an expert analyst. Write a clear summary of the document "{filename}".\n'
    'Use exactly this format:\n\n'
    '## Overview\n'
    '(2-3 sentences: what the document is about)\n\n'
    '## Key Points\n'
    '(5-8 bullet points, keeping the important facts and numbers)\n\n'
    '## Conclusion\n'
    '(1-2 sentences)\n\n'
    'DOCUMENT:\n{text}\n\n'
    'SUMMARY:'
)

PART_PROMPT = ChatPromptTemplate.from_template(
    'Summarise this part of a longer document in 4-6 bullet points. '
    'Keep names, numbers and dates. Output only the bullet points.\n\n'
    'PART:\n{text}\n\nBULLET POINTS:'
)

# Prompt -> model. No output parser because we want raw chunks to stream
summary_chain = SUMMARY_PROMPT | llm
part_chain    = PART_PROMPT | llm | StrOutputParser()


def _split_text(text, size):
    """Cut text into pieces of about `size` characters, preferring paragraph breaks."""
    parts, current = [], ''
    for para in text.split('\n\n'):
        if len(current) + len(para) > size and current:
            parts.append(current)
            current = ''
        while len(para) > size:            # a single giant paragraph
            parts.append(para[:size])
            para = para[size:]
        current += para + '\n\n'
    if current.strip():
        parts.append(current)
    return parts


def stream_summary(documents, filename):
    """
    Sends the whole document to the model and YIELDS the summary as it is
    written, so the browser can show it appearing live.

    Yields : {'type': 'token', 'text': 'The report ...'}
             {'type': 'status', 'text': 'Reading part 2 of 5 ...'}  (long documents only)
    """
    full_text = '\n\n'.join(doc.page_content for doc in documents)

    if len(full_text) > config.SUMMARY_DIRECT_LIMIT:
        parts = _split_text(full_text, config.SUMMARY_PART_SIZE)
        notes = []
        for i, part in enumerate(parts, 1):
            yield {'type': 'status', 'text': 'Reading part {} of {} ...'.format(i, len(parts))}
            notes.append(part_chain.invoke({'text': part}))
        yield {'type': 'status', 'text': 'Writing the final summary ...'}
        full_text = '\n\n'.join(notes)

    for chunk in summary_chain.stream({'filename': filename, 'text': full_text}):
        if chunk.content:
            yield {'type': 'token', 'text': chunk.content}
