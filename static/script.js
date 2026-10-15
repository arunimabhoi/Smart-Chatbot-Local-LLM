// =====================================================================
// Smart Chatbot - browser side
// Sends messages to Flask, reads the streamed NDJSON answer line by line
// and paints it into the chat as it arrives.
// =====================================================================

const $ = (id) => document.getElementById(id);

// ---------- one session id per browser tab ----------
function makeId() {
  return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, '');
}
let SESSION_ID;
try {
  SESSION_ID = sessionStorage.getItem('sc_session') || makeId();
  sessionStorage.setItem('sc_session', SESSION_ID);
} catch (e) { SESSION_ID = makeId(); }

// ---------- modes ----------
const MODES = {
  chat:      { title: 'Chat',            hint: 'General assistant with memory of this conversation.', placeholder: 'Ask anything…' },
  docs:      { title: 'Ask documents',   hint: 'Answers only from the files you uploaded (RAG + FAISS).', placeholder: 'Ask a question about your documents…' },
  agent:     { title: 'Web Agent',       hint: 'Can use web search, Wikipedia, the clock and a calculator.', placeholder: 'e.g. Current gold price in India?' },
  email:     { title: 'Email writer',    hint: 'Describe the email you need. Pick a tone on the left.', placeholder: 'e.g. Ask HR about the interview schedule',
               options: ['professional', 'friendly', 'formal', 'apologetic', 'persuasive', 'sarcastic'] },
  translate: { title: 'Translate',       hint: 'Paste text and choose the language.', placeholder: 'Paste the text to translate…',
               options: ['Hindi', 'Odia', 'Telugu', 'Bengali', 'Tamil', 'French', 'Spanish', 'German', 'English'] },
  explain:   { title: 'Explain simply',  hint: 'Explains a topic for the chosen audience.', placeholder: 'e.g. Gradient descent',
               options: ['beginner', '10 year old kid', '15 year old student', 'graduate', 'interviewer', 'expert'] },
  sentiment: { title: 'Sentiment',       hint: 'Zero-shot sentiment + few-shot emotion label.', placeholder: 'Paste a review or message…' },
};
let mode = 'chat';
let busy = false;
let controller = null;

function setMode(m) {
  mode = m;
  document.querySelectorAll('#modes button').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
  $('modeTitle').textContent = MODES[m].title;
  $('modeHint').textContent = MODES[m].hint;
  $('input').placeholder = MODES[m].placeholder;
  const sel = $('option');
  if (MODES[m].options) {
    sel.innerHTML = MODES[m].options.map(o => `<option>${o}</option>`).join('');
    sel.classList.remove('hidden');
  } else {
    sel.classList.add('hidden');
  }
}

// ---------- tiny, safe markdown renderer ----------
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
}
function renderMarkdown(src) {
  const parts = src.split(/```/);
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {                              // code block
      const body = part.replace(/^[\w+-]*\n/, '');
      html += `<pre><code>${escapeHtml(body)}</code></pre>`;
      return;
    }
    let list = null;
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    escapeHtml(part).split('\n').forEach(line => {
      let m;
      if ((m = line.match(/^\s*#{1,6}\s+(.*)/))) { close(); html += `<h3>${inline(m[1])}</h3>`; }
      else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { if (list !== 'ul') { close(); html += '<ul>'; list = 'ul'; } html += `<li>${inline(m[1])}</li>`; }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== 'ol') { close(); html += '<ol>'; list = 'ol'; } html += `<li>${inline(m[1])}</li>`; }
      else if (line.trim() === '') { close(); }
      else { close(); html += `<p>${inline(line)}</p>`; }
    });
    close();
  });
  return html;
}

// ---------- chat bubbles ----------
function hideWelcome() { const w = $('welcome'); if (w) w.remove(); }

function scrollDown() { const m = $('messages'); m.scrollTop = m.scrollHeight; }

function addUser(text) {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `<div class="bubble"></div>`;
  row.querySelector('.bubble').textContent = text;
  $('messages').appendChild(row);
  scrollDown();
}

function addBot(tagText) {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'msg bot';
  row.innerHTML = `<div class="bubble">
      <div class="tag">${escapeHtml(tagText)}</div>
      <div class="status-line"></div>
      <div class="body cursor"></div>
      <details class="steps hidden"><summary>Agent steps</summary><div class="list"></div></details>
      <details class="sources hidden"><summary>Sources</summary><div class="list"></div></details>
    </div>`;
  $('messages').appendChild(row);
  scrollDown();
  return row.querySelector('.bubble');
}

// ---------- streaming ----------
async function streamInto(bubble, url, payload) {
  const body = bubble.querySelector('.body');
  const statusLine = bubble.querySelector('.status-line');
  let text = '';
  busy = true; toggleBusy(true);
  controller = new AbortController();

  statusLine.textContent = 'Thinking…';
  try {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'token') {
          statusLine.textContent = '';
          text += ev.text;
          body.innerHTML = renderMarkdown(text);
        } else if (ev.type === 'status') {
          statusLine.textContent = ev.text;
        } else if (ev.type === 'step') {
          statusLine.textContent = `Using ${ev.tool}…`;
          addStep(bubble, `<b>🔧 ${escapeHtml(ev.tool)}</b> <code>${escapeHtml(JSON.stringify(ev.input))}</code>`);
        } else if (ev.type === 'step_result') {
          addStep(bubble, `<pre>${escapeHtml(ev.result)}</pre>`);
          statusLine.textContent = 'Reading the result…';
        } else if (ev.type === 'sources') {
          showSources(bubble, ev.sources);
        } else if (ev.type === 'error') {
          body.innerHTML += `<p class="error">${escapeHtml(ev.text)}</p>`;
        }
        scrollDown();
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') body.innerHTML += '<p class="status-line">Stopped.</p>';
    else body.innerHTML += `<p class="error">${escapeHtml(e.message)}</p>`;
  } finally {
    statusLine.textContent = '';
    body.classList.remove('cursor');
    busy = false; toggleBusy(false);
    controller = null;
    scrollDown();
  }
}

function addStep(bubble, html) {
  const d = bubble.querySelector('details.steps');
  d.classList.remove('hidden');
  const el = document.createElement('div');
  el.className = 'step';
  el.innerHTML = html;
  d.querySelector('.list').appendChild(el);
}

function showSources(bubble, sources) {
  if (!sources || !sources.length) return;
  const d = bubble.querySelector('details.sources');
  d.classList.remove('hidden');
  d.querySelector('.list').innerHTML = sources.map(s =>
    `<div class="source"><b>${escapeHtml(s.source || '')}</b> · page ${escapeHtml(String(s.page ?? '-'))}<br>${escapeHtml(s.preview || '')}…</div>`
  ).join('');
}

function toggleBusy(on) {
  $('send').classList.toggle('hidden', on);
  $('stop').classList.toggle('hidden', !on);
  document.querySelectorAll('.file-list button').forEach(b => b.disabled = on);
}

// ---------- send a message ----------
function send() {
  const input = $('input');
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = ''; autoGrow();
  addUser(text);
  const option = MODES[mode].options ? $('option').value : '';
  const tag = MODES[mode].title + (option ? ` · ${option}` : '');
  const bubble = addBot(tag);
  streamInto(bubble, '/api/chat', { session_id: SESSION_ID, message: text, mode, option });
}

function summarize(filename) {
  if (busy) return;
  addUser(`Summarize ${filename}`);
  const bubble = addBot('Summary');
  streamInto(bubble, '/api/summarize', { session_id: SESSION_ID, filename });
  closeSidebarOnPhone();
}

// ---------- uploads ----------
async function upload(files) {
  if (!files || !files.length) return;
  const msg = $('uploadMsg');
  msg.className = 'upload-msg';
  msg.textContent = `Reading and indexing ${files.length} file(s)… (first time can take a minute)`;
  const fd = new FormData();
  fd.append('session_id', SESSION_ID);
  [...files].forEach(f => fd.append('files', f));
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (res.status === 413) throw new Error('File is too large.');
    const data = await res.json();
    renderFiles(data.files || []);
    if (data.errors && data.errors.length) {
      msg.className = 'upload-msg err';
      msg.textContent = data.errors.join(' · ');
    } else {
      msg.textContent = `Added: ${data.added.join(', ')}. Ask about it in “Ask documents” mode, or click Summarize.`;
      setMode('docs');
    }
  } catch (e) {
    msg.className = 'upload-msg err';
    msg.textContent = 'Upload failed: ' + e.message;
  }
  $('fileInput').value = '';
}

function renderFiles(files) {
  $('fileList').innerHTML = files.map(f => `
    <li>
      <div class="file-name">${escapeHtml(f.name)}</div>
      <div class="file-meta">${f.pages} page(s) · ${f.chunks} chunks</div>
      <button data-sum="${escapeHtml(f.name)}">Summarize</button>
      <button data-ask="1">Ask</button>
    </li>`).join('');
}

// ---------- health ----------
async function checkHealth() {
  const dot = $('statusDot'), txt = $('statusText');
  try {
    const h = await (await fetch('/api/health')).json();
    if (!h.running) { dot.className = 'dot bad'; txt.textContent = 'Ollama not running — run: ollama serve'; }
    else if (h.missing.length) { dot.className = 'dot bad'; txt.textContent = 'Missing: ' + h.missing.map(m => 'ollama pull ' + m).join(' · '); }
    else { dot.className = 'dot ok'; txt.textContent = 'Ollama connected'; }
  } catch (e) { dot.className = 'dot bad'; txt.textContent = 'Server not reachable'; }
}

// ---------- reset ----------
async function reset(keepDocuments) {
  if (busy) return;
  const res = await fetch('/api/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: SESSION_ID, keep_documents: keepDocuments }),
  });
  const data = await res.json();
  renderFiles(data.files || []);
  if (!keepDocuments) $('uploadMsg').textContent = '';
  location.hash = '';
  $('messages').innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome'; w.id = 'welcome';
  w.innerHTML = `<h1>New conversation</h1><p>${keepDocuments ? 'Your uploaded documents are still available.' : 'Documents cleared.'}</p>`;
  $('messages').appendChild(w);
}

// ---------- misc UI ----------
function autoGrow() { const t = $('input'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 180) + 'px'; }
function closeSidebarOnPhone() { $('sidebar').classList.remove('open'); }

// ---------- wire everything up ----------
document.querySelectorAll('#modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('send').addEventListener('click', send);
$('stop').addEventListener('click', () => controller && controller.abort());
$('input').addEventListener('input', autoGrow);
$('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
$('fileInput').addEventListener('change', e => upload(e.target.files));
$('newChat').addEventListener('click', () => reset(true));
$('clearAll').addEventListener('click', () => reset(false));
$('menuBtn').addEventListener('click', () => $('sidebar').classList.toggle('open'));

$('fileList').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.sum) summarize(b.dataset.sum);
  if (b.dataset.ask) { setMode('docs'); closeSidebarOnPhone(); $('input').focus(); }
});

const dz = $('dropzone');
['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', e => upload(e.dataTransfer.files));

document.addEventListener('click', e => {
  const card = e.target.closest('.card');
  if (!card) return;
  setMode(card.dataset.mode);
  if (card.dataset.prompt) { $('input').value = card.dataset.prompt; autoGrow(); }
  $('input').focus();
});

setMode('chat');
checkHealth();
setInterval(checkHealth, 20000);
