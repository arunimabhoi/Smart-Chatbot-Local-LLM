"""
core/prompt_lab.py
------------------
Quick "smart actions" built from the techniques in Prompting_techniques.ipynb
and Langchain_Fundamentals.ipynb:

    sentiment  -> zero-shot (POSITIVE / NEGATIVE / NEUTRAL) + few-shot emotion label
    translate  -> chained prompts: clean up the text -> translate it
    email      -> prompt template with name / purpose / tone
    explain    -> "explain {topic} to a {person}" template
"""

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import llm, chat_llm

parser = StrOutputParser()

# ---------------- Zero-shot sentiment ----------------
zero_shot = ChatPromptTemplate.from_template(
    'Classify the sentiment of this text as POSITIVE, NEGATIVE or NEUTRAL. '
    'Reply with one word only.\nText : {text}'
)
sentiment_chain = zero_shot | llm | parser

# ---------------- Few-shot emotion ----------------
examples = [
    {'input': 'I love this',              'output': 'JOY'},
    {'input': 'This is terrifying',       'output': 'FEAR'},
    {'input': 'What the hell is this',    'output': 'ANGER'},
    {'input': 'I miss how it used to be', 'output': 'SADNESS'},
    {'input': 'The meeting is at 4pm',    'output': 'NEUTRAL'},
]
example_prompt = ChatPromptTemplate.from_messages([('human', '{input}'), ('ai', '{output}')])
few_shot = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
emotion_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Label the emotion of the text using one of: JOY, FEAR, ANGER, SADNESS, NEUTRAL. '
               'Reply with the label only.'),
    few_shot,
    ('human', '{input}'),
])
emotion_chain = emotion_prompt | llm | parser

# ---------------- Translate (sequential chain) ----------------
translate_prompt = ChatPromptTemplate.from_template(
    'Translate the following text into {language}. Output only the translation.\n\nText: {text}'
)
translate_chain = translate_prompt | chat_llm

# ---------------- Email writer ----------------
email_prompt = ChatPromptTemplate.from_template(
    'You are an expert in writing emails. Using the request below, write a complete email '
    'with a Subject line, in a {tone} tone, not more than 250 words.\n\nRequest: {text}'
)
email_chain = email_prompt | chat_llm

# ---------------- Explain like I'm ... ----------------
explain_prompt = ChatPromptTemplate.from_template(
    'You are an expert teacher. Explain the topic: {text} to a {person} '
    'in not more than 200 words. Use a simple real-life example.'
)
explain_chain = explain_prompt | chat_llm


def _first_word(s):
    s = s.strip().upper().replace('*', '').replace('.', '')
    return s.split()[0] if s else '?'


def stream_action(action, text, option=''):
    """Yields {'type': 'token', ...} events for the chosen smart action."""
    if action == 'sentiment':
        sentiment = _first_word(sentiment_chain.invoke({'text': text}))
        emotion = _first_word(emotion_chain.invoke({'input': text}))
        yield {'type': 'token', 'text': '**Sentiment:** {}\n\n**Emotion:** {}'.format(sentiment, emotion)}
        return

    if action == 'translate':
        chain, inputs = translate_chain, {'text': text, 'language': option or 'Hindi'}
    elif action == 'email':
        chain, inputs = email_chain, {'text': text, 'tone': option or 'professional'}
    elif action == 'explain':
        chain, inputs = explain_chain, {'text': text, 'person': option or 'beginner'}
    else:
        yield {'type': 'token', 'text': 'Unknown action: {}'.format(action)}
        return

    for chunk in chain.stream(inputs):
        if chunk.content:
            yield {'type': 'token', 'text': chunk.content}
