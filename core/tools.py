"""
core/tools.py
-------------
Tools the agent is allowed to use.                         (AI_Agent.ipynb)
All of them are FREE - no API keys:
    get_datetime  -> your computer's clock
    wikipedia     -> Wikipedia's public API
    web_search    -> DuckDuckGo (ddgs package)
    calculator    -> safe maths in Python (small models are bad at arithmetic)
"""

import ast
import datetime
import math
import operator

import requests
from langchain_core.tools import tool


# Tool 1 : Get current date and time
@tool
def get_datetime(query: str = '') -> str:
    """
    Returns the current date and time.
    Use this when the user asks about the current date, time, year,
    today's date or the day of the week. Input can be an empty string.
    """
    now = datetime.datetime.now()
    return 'Current date : {}\nCurrent time : {}\nCurrent year : {}'.format(
        now.strftime('%A, %B %d, %Y'), now.strftime('%I:%M %p'), now.year)


# Tool 2 : Web Search (DuckDuckGo)
@tool
def web_search(query: str) -> str:
    """
    Searches the web for current, recent, or real-time information.
    Use this for news, prices, recent events, or anything that might have changed recently.
    Input should be a clear search query.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().invoke(query)[:2000]
    except Exception as e:
        return 'Web search failed: {}'.format(e)


# Tool 3 : Wikipedia Search
@tool
def wikipedia(query: str) -> str:
    """
    Searches Wikipedia for factual information about people, places, events,
    concepts, history, science, and general knowledge.
    Use this for well-established facts, biographies, or encyclopedic information.
    Input should be a concise search term.
    """
    headers = {'User-Agent': 'SmartChatbot/1.0 (LangChain learning project)'}
    try:
        # 1) Find the best-matching article title
        search = requests.get('https://en.wikipedia.org/w/rest.php/v1/search/page',
                              params={'q': query, 'limit': 1}, headers=headers, timeout=10)
        search.raise_for_status()
        pages = search.json().get('pages', [])
        if not pages:
            return "No Wikipedia article found for '{}'.".format(query)
        title = pages[0]['title']

        # 2) Fetch a plain-text summary of that article
        title_url = requests.utils.quote(title.replace(' ', '_'))
        summary = requests.get('https://en.wikipedia.org/api/rest_v1/page/summary/' + title_url,
                               headers=headers, timeout=10)
        summary.raise_for_status()
        extract = summary.json().get('extract', '')
        if not extract:
            return "No summary available for '{}'.".format(title)
        return '{}: {}'.format(title, extract)[:1200]
    except Exception as e:
        return 'Wikipedia lookup failed: {}'.format(e)


# Tool 4 : Calculator (safe - no eval of arbitrary code)
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {'sqrt': math.sqrt, 'round': round, 'abs': abs, 'log': math.log, 'pi': math.pi}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _FUNCS:
        return _FUNCS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError('unsupported expression')


@tool
def calculator(expression: str) -> str:
    """
    Calculates a maths expression exactly, e.g. "5*32 - 78" or "sqrt(144) + 2**3".
    Use this for ANY arithmetic instead of calculating in your head.
    """
    try:
        expr = expression.replace('^', '**').replace(',', '')
        result = _eval(ast.parse(expr, mode='eval').body)
        return '{} = {}'.format(expression, result)
    except Exception as e:
        return 'Could not calculate "{}": {}'.format(expression, e)


ALL_TOOLS = [get_datetime, web_search, wikipedia, calculator]
