"""
core/agent.py
-------------
Two ways of answering a normal chat message:

1. stream_chat()   - plain chat with memory          (app_history.py)
2. stream_agent()  - an AI agent that can decide to  (AI_Agent.ipynb)
                     call tools (web, Wikipedia, clock, calculator)
"""

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from core.llm import chat_llm, agent_llm
from core.tools import ALL_TOOLS


# ------------------------------------------------------------------
# 1) Plain chat with history
# ------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = (
    'You are Smart Chatbot, a friendly and knowledgeable AI assistant running locally '
    'on the user\'s computer. Answer clearly and concisely. Start with a direct answer, '
    'then explain. Use bullet points or short code blocks when they help. '
    'For maths, reason step by step. If you are not sure about something, say so '
    'instead of guessing. You do not have internet access in this mode - if the user '
    'needs live information (news, prices, today\'s date), tell them to switch to '
    '"Web Agent" mode.'
)


def to_messages(history):
    """Convert [{'role': 'user', 'content': ...}, ...] into LangChain message objects."""
    out = []
    for m in history:
        if m['role'] == 'user':
            out.append(HumanMessage(content=m['content']))
        elif m['role'] == 'assistant':
            out.append(AIMessage(content=m['content']))
    return out


def stream_chat(message, history):
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + to_messages(history)
    messages.append(HumanMessage(content=message))
    for chunk in chat_llm.stream(messages):
        if chunk.content:
            yield {'type': 'token', 'text': chunk.content}


# ------------------------------------------------------------------
# 2) Agent with tools
# ------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = (
    'You are a smart, helpful AI assistant with access to tools. '
    'Use a tool only when it is actually needed: '
    'get_datetime for the current date/time, web_search for recent or live information '
    '(news, prices, scores), wikipedia for well-known facts and biographies, '
    'calculator for any arithmetic. Otherwise answer directly. '
    'When you have enough information, give a clear, concise final answer.'
)

agent = create_agent(
    model=agent_llm,
    tools=ALL_TOOLS,
    system_prompt=AGENT_SYSTEM_PROMPT,
)


def stream_agent(message, history):
    """
    Runs the agent and yields:
        {'type': 'step', 'tool': 'wikipedia', 'input': {...}}        - tool chosen
        {'type': 'step_result', 'tool': 'wikipedia', 'result': '..'} - what it returned
        {'type': 'token', 'text': '...'}                             - final answer
    Same idea as run_agent(show_steps=True) in the notebook, but live.
    """
    msgs = [{'role': m['role'], 'content': m['content']} for m in history]
    msgs.append({'role': 'user', 'content': message})

    answered = False
    for update in agent.stream({'messages': msgs}, stream_mode='updates'):
        for node_output in update.values():
            if not isinstance(node_output, dict):
                continue
            for m in node_output.get('messages', []):
                if isinstance(m, ToolMessage):
                    yield {'type': 'step_result', 'tool': m.name,
                           'result': str(m.content)[:400]}
                elif isinstance(m, AIMessage):
                    if m.tool_calls:
                        for call in m.tool_calls:
                            yield {'type': 'step', 'tool': call['name'], 'input': call['args']}
                    elif m.content:
                        answered = True
                        yield {'type': 'token', 'text': m.content}

    if not answered:
        yield {'type': 'token',
               'text': 'The agent finished without an answer. Try rephrasing, or use a larger model.'}
