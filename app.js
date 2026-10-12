/**
 * JKBOSE AI — ChatGPT / Gemini Style Frontend Application
 * Handles collapsible sidebar, multi-turn chat, markdown rendering,
 * speech recognition/synthesis, question history, and paper explorer.
 */

const API_BASE = (window.location.protocol.startsWith('http')) ? `${window.location.origin}/api` : 'http://localhost:5000/api';

// ─── Application State ──────────────────────────────────────────────
let currentView = 'chat';              // 'chat' | 'explorer'
let currentMode = 'exam_answer';       // 'exam_answer' | 'tutor' | 'trends' | 'quiz'
let currentEngine = 'fast';            // 'fast' | 'ollama'
let sessionId = 'student_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
let speechRecognition = null;
let isListening = false;

// Mode Label Lookup
const MODE_NAMES = {
    exam_answer: '📘 Model Exam Answer',
    tutor: '💡 Concept Tutor',
    trends: '📊 Past Repeats & Trends',
    quiz: '🎯 Practice Quiz'
};

// ─── DOM Elements ───────────────────────────────────────────────────
// Sidebar
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');
const btnNewChat = document.getElementById('btnNewChat');
const sidebarModes = document.getElementById('sidebarModes');
const chatClass = document.getElementById('chatClass');
const chatSubject = document.getElementById('chatSubject');
const btnEngineFast = document.getElementById('btnEngineFast');
const btnEngineOllama = document.getElementById('btnEngineOllama');
const ollamaBadge = document.getElementById('ollamaBadge');
const ollamaText = document.getElementById('ollamaText');
const historyList = document.getElementById('historyList');
const btnClearHistory = document.getElementById('btnClearHistory');
const btnSwitchToExplorer = document.getElementById('btnSwitchToExplorer');
const btnStats = document.getElementById('btnStats');

// Top Bar
const currentModeBadge = document.getElementById('currentModeBadge');
const currentEngineBadge = document.getElementById('currentEngineBadge');
const btnViewToggle = document.getElementById('btnViewToggle');
const viewToggleText = document.getElementById('viewToggleText');
const btnResetSession = document.getElementById('btnResetSession');

// Chat View
const chatViewArea = document.getElementById('chatViewArea');
const chatScrollContainer = document.getElementById('chatScrollContainer');
const chatHero = document.getElementById('chatHero');
const messageStream = document.getElementById('messageStream');
const chatTyping = document.getElementById('chatTyping');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const btnVoice = document.getElementById('btnVoice');

// Explorer View
const explorerViewArea = document.getElementById('explorerViewArea');
const filterClass = document.getElementById('filterClass');
const filterSubject = document.getElementById('filterSubject');
const filterYear = document.getElementById('filterYear');
const filterMarks = document.getElementById('filterMarks');
const searchInput = document.getElementById('searchInput');
const btnSearch = document.getElementById('btnSearch');
const explorerEmpty = document.getElementById('explorerEmpty');
const explorerLoading = document.getElementById('explorerLoading');
const explorerContent = document.getElementById('explorerContent');

// Stats Modal & Toast
const statsModal = document.getElementById('statsModal');
const btnCloseStats = document.getElementById('btnCloseStats');
const statsBody = document.getElementById('statsBody');
const toast = document.getElementById('toast');

// ─── App Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Always start with a completely fresh, clean chat
    if (chatInput) {
        chatInput.value = '';
        chatInput.style.height = 'auto';
    }
    if (searchInput) searchInput.value = '';
    if (messageStream) messageStream.innerHTML = '';
    if (chatHero) chatHero.style.display = 'flex';

    initSidebar();
    initChatInput();
    initSubjectChips();
    initKeyboardShortcuts();
    initSpeechRecognition();
    initExplorer();
    initStatsModal();
    checkOllamaStatus();
    renderHistoryList();
});

// ─── Sidebar Logic ──────────────────────────────────────────────────
function initSidebar() {
    // Sidebar toggle (desktop collapse & mobile drawer)
    btnToggleSidebar.addEventListener('click', () => {
        if (window.innerWidth <= 900) {
            sidebar.classList.toggle('open');
            sidebarOverlay.classList.toggle('active');
        } else {
            sidebar.classList.toggle('collapsed');
        }
    });

    sidebarOverlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    });

    // New Chat Button
    btnNewChat.addEventListener('click', startNewChat);

    // Study Modes in Sidebar
    sidebarModes.querySelectorAll('.sidebar-item').forEach(item => {
        item.addEventListener('click', () => {
            sidebarModes.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            currentMode = item.dataset.mode;
            currentModeBadge.textContent = MODE_NAMES[currentMode] || '📘 Model Exam Answer';
            showToast(`Mode switched to: ${item.querySelector('.item-name').textContent}`);

            // If on mobile, close sidebar after selecting mode
            if (window.innerWidth <= 900) {
                sidebar.classList.remove('open');
                sidebarOverlay.classList.remove('active');
            }
        });
    });

    // Engine Selector Pills
    btnEngineFast.addEventListener('click', () => {
        btnEngineFast.classList.add('active');
        btnEngineOllama.classList.remove('active');
        currentEngine = 'fast';
        currentEngineBadge.textContent = '⚡ Fast Solver';
        showToast('Switched to Fast Exam Engine (<0.5s)');
    });

    btnEngineOllama.addEventListener('click', () => {
        btnEngineOllama.classList.add('active');
        btnEngineFast.classList.remove('active');
        currentEngine = 'ollama';
        currentEngineBadge.textContent = '🤖 Deep AI (Ollama)';
        showToast('Switched to Deep AI (Ollama)');
    });

    // Clear History Button
    btnClearHistory.addEventListener('click', () => {
        localStorage.removeItem('jkbose_recent_queries');
        renderHistoryList();
        showToast('Recent question history cleared');
    });

    // Switch to Explorer from Sidebar
    btnSwitchToExplorer.addEventListener('click', () => {
        toggleView('explorer');
        if (window.innerWidth <= 900) {
            sidebar.classList.remove('open');
            sidebarOverlay.classList.remove('active');
        }
    });

    // Top Bar View Toggle
    btnViewToggle.addEventListener('click', () => {
        toggleView(currentView === 'chat' ? 'explorer' : 'chat');
    });

    // Reset current chat session
    btnResetSession.addEventListener('click', startNewChat);
}

function toggleView(targetView) {
    currentView = targetView;
    if (targetView === 'explorer') {
        chatViewArea.style.display = 'none';
        explorerViewArea.style.display = 'block';
        viewToggleText.textContent = '💬 Back to Chatbot';
        searchInput.focus();
    } else {
        explorerViewArea.style.display = 'none';
        chatViewArea.style.display = 'flex';
        viewToggleText.textContent = '🔍 Paper Explorer';
        chatInput.focus();
    }
}

function startNewChat() {
    // Generate fresh session ID
    sessionId = 'student_' + Math.random().toString(36).substring(2, 9);
    localStorage.setItem('jkbose_chat_session', sessionId);

    // Clear message DOM & show hero
    messageStream.innerHTML = '';
    chatHero.style.display = 'flex';
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // If on explorer view, return to chat
    if (currentView !== 'chat') toggleView('chat');

    chatInput.focus();
    showToast('Started a fresh chat session');
}

// ─── Chat Input & Messaging ─────────────────────────────────────────
function initChatInput() {
    // Hero suggestion cards
    document.querySelectorAll('.hero-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.dataset.prompt;
            chatInput.value = prompt;
            submitChatMessage();
        });
    });

    // Textarea input sizing and send button highlight
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
        if (chatInput.value.trim().length > 0) {
            btnSend.classList.add('active');
        } else {
            btnSend.classList.remove('active');
        }
    });

    // Enter to submit, Shift+Enter for new line
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitChatMessage();
        }
    });

    btnSend.addEventListener('click', submitChatMessage);
}

// ─── Subject Chips Filter Selection ────────────────────────────────
function initSubjectChips() {
    const chips = document.querySelectorAll('.subject-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const subject = chip.dataset.subject;
            const wasActive = chip.classList.contains('active');

            chips.forEach(c => c.classList.remove('active'));

            if (!wasActive) {
                chip.classList.add('active');
                if (chatSubject) chatSubject.value = subject;
                showToast(`Filtered for ${subject} questions`);
            } else {
                if (chatSubject) chatSubject.value = '';
                showToast(`Cleared subject filter (All Subjects)`);
            }
        });
    });

    if (chatSubject) {
        chatSubject.addEventListener('change', () => {
            const val = chatSubject.value;
            chips.forEach(c => {
                if (c.dataset.subject === val) {
                    c.classList.add('active');
                } else {
                    c.classList.remove('active');
                }
            });
        });
    }
}

// ─── Keyboard Shortcuts ────────────────────────────────────────────
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            startNewChat();
        }
    });
}

// ─── Submit Chat Message ────────────────────────────────────────────
async function submitChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.classList.remove('active');

    // Hide welcome hero on first message
    chatHero.style.display = 'none';

    // Append User Message
    appendUserBubble(text);
    scrollChatBottom();

    // Show Typing Indicator
    chatTyping.style.display = 'flex';
    scrollChatBottom();

    // Save query to Sidebar Recent Questions
    saveToRecentHistory(text);

    // Build Filters
    const filters = {};
    if (chatClass.value) filters.class_level = chatClass.value;
    if (chatSubject.value) filters.subject = chatSubject.value;

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: sessionId,
                mode: currentMode,
                engine_type: currentEngine,
                filters: filters
            })
        });

        const data = await res.json();
        chatTyping.style.display = 'none';

        if (data.error) {
            appendAssistantBubble({
                reply: `⚠️ **Error:** ${data.error}`,
                mode: currentMode,
                engine_used: 'error',
                confidence: 'NONE',
                sources: [],
                suggested_followups: ["Try asking again", "State Ohm's Law"]
            });
        } else {
            appendAssistantBubble(data);
        }
    } catch (err) {
        chatTyping.style.display = 'none';
        appendAssistantBubble({
            reply: `⚠️ **Connection Error:** Could not connect to the JKBOSE server at \`http://localhost:5000\`. Please ensure the backend is active.`,
            mode: currentMode,
            engine_used: 'offline',
            confidence: 'NONE',
            sources: [],
            suggested_followups: ["Retry question", "Check server health"]
        });
    }

    scrollChatBottom();
}

// ─── Message Bubbles (ChatGPT/Gemini Style) ──────────────────────────
function appendUserBubble(text) {
    const row = document.createElement('div');
    row.className = 'chat-message-row user';
    row.innerHTML = `
        <div class="msg-bubble-box">
            ${escapeHtml(text)}
        </div>
        <div class="msg-avatar-icon" title="You">🧑‍🎓</div>
    `;
    messageStream.appendChild(row);
}

function appendAssistantBubble(data) {
    const row = document.createElement('div');
    row.className = 'chat-message-row assistant';

    const modeDisplay = MODE_NAMES[data.mode] || '📘 Exam Solution';
    const confidence = data.confidence || 'HIGH';
    const engineDisplay = data.engine_used || currentEngine;

    // Citations drawer
    let citationsHtml = '';
    if (data.sources && data.sources.length > 0) {
        citationsHtml = `
            <div class="msg-citations-drawer">
                <button class="citations-toggle" onclick="toggleCitations(this)">
                    <span>📑 <strong>${data.sources.length}</strong> JKBOSE Past Paper Questions Cited</span>
                    <span class="chevron">▼</span>
                </button>
                <div class="citations-list" style="display:none;">
                    ${data.sources.map(s => `
                        <div class="citation-card">
                            <div class="citation-badges">
                                <span class="citation-tag">${s.year || 'Exam'}</span>
                                <span class="citation-tag">Class ${s.class_level || '10/12'}</span>
                                <span class="citation-tag">${s.subject || 'Subject'}</span>
                                ${s.marks ? `<span class="citation-tag">${s.marks}M</span>` : ''}
                                ${s.paper_code ? `<span class="citation-tag">Code: ${s.paper_code}</span>` : ''}
                            </div>
                            <div class="citation-snippet">"${escapeHtml(s.text)}"</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // Follow-up chips
    let followupsHtml = '';
    if (data.suggested_followups && data.suggested_followups.length > 0) {
        followupsHtml = `
            <div class="followups-wrapper">
                <span class="followups-label">Suggested follow-ups:</span>
                <div class="followups-list">
                    ${data.suggested_followups.map(f => `
                        <button class="followup-pill" onclick="sendFollowup(this)">${escapeHtml(f)}</button>
                    `).join('')}
                </div>
            </div>
        `;
    }

    row.innerHTML = `
        <div class="msg-avatar-icon" title="JKBOSE AI">🎓</div>
        <div class="msg-bubble-box">
            <div class="msg-top-meta">
                <div class="meta-tags">
                    <span class="badge-mode-pill">${modeDisplay}</span>
                    <span class="badge-engine-pill">${escapeHtml(engineDisplay)}</span>
                </div>
            </div>

            <div class="msg-body-markdown">
                ${parseMarkdown(data.reply)}
            </div>

            ${citationsHtml}

            <div class="msg-bottom-bar">
                <button class="action-btn" onclick="copySolution(this)" title="Copy solution to clipboard">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    <span>Copy</span>
                </button>
                <button class="action-btn" onclick="speakSolution(this)" title="Listen via speech synthesis">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
                    <span>Read Aloud</span>
                </button>
            </div>

            ${followupsHtml}
        </div>
    `;

    messageStream.appendChild(row);
}

// ─── Actions on Message ─────────────────────────────────────────────
window.toggleCitations = function(btn) {
    const list = btn.nextElementSibling;
    const chevron = btn.querySelector('.chevron');
    if (list.style.display === 'none') {
        list.style.display = 'flex';
        chevron.textContent = '▲';
    } else {
        list.style.display = 'none';
        chevron.textContent = '▼';
    }
};

window.copySolution = function(btn) {
    const bubble = btn.closest('.msg-bubble-box');
    const body = bubble.querySelector('.msg-body-markdown');
    const text = body.innerText || body.textContent;

    navigator.clipboard.writeText(text).then(() => {
        showToast('📋 Copied solution to clipboard!');
        const prevHtml = btn.innerHTML;
        btn.innerHTML = '<span>✅ Copied</span>';
        setTimeout(() => { btn.innerHTML = prevHtml; }, 2000);
    }).catch(() => {
        showToast('Failed to copy text');
    });
};

window.speakSolution = function(btn) {
    if (!('speechSynthesis' in window)) {
        showToast('Text-to-speech not supported');
        return;
    }

    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        btn.querySelector('span').textContent = 'Read Aloud';
        return;
    }

    const bubble = btn.closest('.msg-bubble-box');
    const body = bubble.querySelector('.msg-body-markdown');
    const cleanText = body.innerText.replace(/[*_#$`~]/g, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
        btn.querySelector('span').textContent = 'Stop';
    };
    utterance.onend = utterance.onerror = () => {
        btn.querySelector('span').textContent = 'Read Aloud';
    };

    window.speechSynthesis.speak(utterance);
};

window.sendFollowup = function(btn) {
    chatInput.value = btn.textContent.trim();
    submitChatMessage();
};

function scrollChatBottom() {
    setTimeout(() => {
        chatScrollContainer.scrollTop = chatScrollContainer.scrollHeight;
    }, 50);
}

// ─── Markdown Parser ────────────────────────────────────────────────
function parseMarkdown(md) {
    if (!md) return '';
    let text = md;

    // Preserve details / summary blocks
    const detailsBlocks = [];
    text = text.replace(/<details>[\s\S]*?<\/details>/gi, (m) => {
        detailsBlocks.push(m);
        return `__DETAILS_BLOCK_${detailsBlocks.length - 1}__`;
    });

    // HTML escape
    text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Headers with smart colored badges
    text = text.replace(/^#### (.*?)$/gm, '<h4>$1</h4>');
    text = text.replace(/^### (Step \d+[:\s].*?)$/gm, (m, p1) => {
        const stepMatch = p1.match(/Step \d+/);
        const stepNum = stepMatch ? stepMatch[0] : 'Step';
        const rest = p1.replace(/Step \d+[:\s]*/, '');
        return `<h3><span class="step-badge">${stepNum}</span> ${rest}</h3>`;
    });
    text = text.replace(/^### (Marking Scheme[:\s]*.*?)$/gm, (m, p1) => {
        return `<h3><span class="marking-badge">Marking Scheme</span></h3>`;
    });
    text = text.replace(/^### (Tip[:\s]*.*?)$/gm, (m, p1) => {
        return `<h3><span class="tip-badge">Exam Tip</span> ${p1.replace(/Tip[:\s]*/, '')}</h3>`;
    });
    text = text.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*?)$/gm, '<h3>$1</h3>');
    text = text.replace(/^# (.*?)$/gm, '<h2>$1</h2>');

    // Bold & Italics
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Math Formulas: $$...$$ and $...$
    text = text.replace(/\$\$(.*?)\$\$/gs, '<div class="math-block"><code>$$$1$$</code></div>');
    text = text.replace(/\$([^\$\n]+?)\$/g, '<code class="math-inline">$1</code>');

    // Blockquotes
    text = text.replace(/^&gt; (.*?)$/gm, '<blockquote>$1</blockquote>');

    // Tables
    text = text.replace(/(\|.+?\|\n)+/g, (match) => {
        const rows = match.trim().split('\n');
        if (rows.length < 2) return match;
        let html = '<table>';
        rows.forEach((row, i) => {
            if (row.includes('---')) return;
            const cells = row.split('|').filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
            if (i === 0) {
                html += '<tr>' + cells.map(c => `<th>${c.trim()}</th>`).join('') + '</tr>';
            } else {
                html += '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
            }
        });
        html += '</table>';
        return html;
    });

    // Lists
    text = text.replace(/^\s*-\s+(.*?)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*?<\/li>(\n|$))+/g, '<ul>$&</ul>');
    text = text.replace(/^\s*\d+\.\s+(.*?)$/gm, '<li>$1</li>');

    // Paragraph line breaks
    text = text.replace(/\n\n/g, '<p></p>');

    // Restore details
    detailsBlocks.forEach((b, i) => {
        text = text.replace(`__DETAILS_BLOCK_${i}__`, b);
    });

    return text;
}

// ─── Voice Recognition (Web Speech API) ─────────────────────────────
function initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        btnVoice.style.opacity = '0.4';
        btnVoice.title = 'Voice input not supported in this browser';
        return;
    }

    speechRecognition = new SpeechRec();
    speechRecognition.continuous = false;
    speechRecognition.interimResults = false;
    speechRecognition.lang = 'en-US';

    speechRecognition.onstart = () => {
        isListening = true;
        btnVoice.classList.add('listening');
        showToast('🎙️ Listening... Speak your question now');
    };

    speechRecognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        chatInput.value = text;
        btnSend.classList.add('active');
        chatInput.focus();
        showToast(`Heard: "${text}"`);
    };

    speechRecognition.onerror = speechRecognition.onend = () => {
        isListening = false;
        btnVoice.classList.remove('listening');
    };

    btnVoice.addEventListener('click', () => {
        if (isListening) {
            speechRecognition.stop();
        } else {
            speechRecognition.start();
        }
    });
}

// ─── Recent Questions History in Sidebar ────────────────────────────
function saveToRecentHistory(query) {
    if (!query || query.length < 3) return;
    let list = JSON.parse(localStorage.getItem('jkbose_recent_queries') || '[]');
    list = list.filter(q => q.toLowerCase() !== query.toLowerCase());
    list.unshift(query);
    if (list.length > 15) list = list.slice(0, 15);
    localStorage.setItem('jkbose_recent_queries', JSON.stringify(list));
    renderHistoryList();
}

function renderHistoryList() {
    const list = JSON.parse(localStorage.getItem('jkbose_recent_queries') || '[]');
    if (!list || list.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No recent questions yet</div>';
        return;
    }

    historyList.innerHTML = list.map(q => `
        <div class="history-item" onclick="loadHistoryItem(this)" title="${escapeHtml(q)}">
            <span>💬 ${escapeHtml(q.length > 28 ? q.substring(0, 28) + '...' : q)}</span>
        </div>
    `).join('');
}

window.loadHistoryItem = function(el) {
    const query = el.getAttribute('title');
    chatInput.value = query;
    if (currentView !== 'chat') toggleView('chat');
    submitChatMessage();

    if (window.innerWidth <= 900) {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    }
};

// ─── Session History Loading ────────────────────────────────────────
async function loadSessionHistory() {
    try {
        const res = await fetch(`${API_BASE}/chat/history?session_id=${sessionId}`);
        const data = await res.json();
        if (data.history && data.history.length > 0) {
            chatHero.style.display = 'none';
            data.history.forEach(turn => {
                if (turn.role === 'user') {
                    appendUserBubble(turn.content);
                } else if (turn.role === 'assistant') {
                    appendAssistantBubble({
                        reply: turn.content,
                        mode: currentMode,
                        engine_used: 'session_cache',
                        confidence: 'HIGH',
                        sources: [],
                        suggested_followups: []
                    });
                }
            });
            scrollChatBottom();
        }
    } catch (e) {
        console.debug('No saved turns in current session');
    }
}

// ─── Check Ollama Status ────────────────────────────────────────────
async function checkOllamaStatus() {
    try {
        const res = await fetch(`${API_BASE}/ollama/status`);
        const data = await res.json();
        if (data.available && data.models.length > 0) {
            ollamaBadge.classList.remove('offline');
            const model = data.active_model || data.models[0];
            ollamaText.textContent = `Ollama: ${model}`;
            ollamaBadge.title = `Models: ${data.models.join(', ')}`;
        } else {
            ollamaBadge.classList.add('offline');
            ollamaText.textContent = 'Ollama Offline';
        }
    } catch (e) {
        ollamaBadge.classList.add('offline');
        ollamaText.textContent = 'Fast Engine Active';
    }
}

// ─── Paper Explorer View Logic ──────────────────────────────────────
function initExplorer() {
    btnSearch.addEventListener('click', performExplorerSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performExplorerSearch();
    });
}

async function performExplorerSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    let fullQuery = query;
    if (filterClass.value) fullQuery += ` class ${filterClass.value}`;
    if (filterSubject.value) fullQuery += ` ${filterSubject.value}`;
    if (filterYear.value) fullQuery += ` ${filterYear.value}`;
    if (filterMarks.value) fullQuery += ` ${filterMarks.value} marks`;

    explorerEmpty.style.display = 'none';
    explorerContent.style.display = 'none';
    explorerLoading.style.display = 'flex';

    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: fullQuery,
                top_k: 25,
                top_n: 10
            })
        });

        const data = await res.json();
        explorerLoading.style.display = 'none';
        renderExplorerResults(data);
    } catch (err) {
        explorerLoading.style.display = 'none';
        explorerContent.style.display = 'block';
        explorerContent.innerHTML = `<p style="color:var(--error); text-align:center;">Failed to connect to server.</p>`;
    }
}

function renderExplorerResults(data) {
    explorerContent.style.display = 'block';
    if (!data.questions || data.questions.length === 0) {
        explorerContent.innerHTML = `<p style="text-align:center; color:var(--text-muted); padding:30px;">No questions matched your search criteria.</p>`;
        return;
    }

    let html = `
        <div style="margin-bottom:14px; font-size:0.85rem; color:var(--text-muted);">
            Found <strong>${data.questions.length}</strong> matching questions
        </div>
    `;

    data.questions.forEach(q => {
        const m = q.metadata;
        html += `
            <div class="question-card">
                <div class="question-card-header">
                    <div class="question-meta">
                        ${m.class_level ? `<span class="meta-tag class-tag">Class ${m.class_level}</span>` : ''}
                        ${m.subject ? `<span class="meta-tag subject-tag">${m.subject}</span>` : ''}
                        ${m.year ? `<span class="meta-tag">${m.year}</span>` : ''}
                        ${m.marks ? `<span class="meta-tag">${m.marks}M</span>` : ''}
                        ${m.question_number ? `<span class="meta-tag">Q.${m.question_number}</span>` : ''}
                    </div>
                    <div class="score-badge">Rel: ${(q.score * 100).toFixed(0)}%</div>
                </div>
                <div class="question-text">${escapeHtml(q.text)}</div>
                <div class="question-citation">${escapeHtml(q.citation)}</div>
            </div>
        `;
    });

    explorerContent.innerHTML = html;
}

// ─── Stats Modal Logic ──────────────────────────────────────────────
function initStatsModal() {
    btnStats.addEventListener('click', async () => {
        statsModal.style.display = 'flex';
        statsBody.innerHTML = '<div class="spinner"></div>';
        try {
            const res = await fetch(`${API_BASE}/statistics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filters: {} })
            });
            const data = await res.json();
            renderStatsBody(data);
        } catch (e) {
            statsBody.innerHTML = `<p style="color:var(--error)">Error loading stats: ${e.message}</p>`;
        }
    });

    btnCloseStats.addEventListener('click', () => {
        statsModal.style.display = 'none';
    });
}

function renderStatsBody(data) {
    statsBody.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${(data.total_questions || 9573).toLocaleString()}</div>
                <div class="stat-label">Total Questions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(data.total_papers || 515).toLocaleString()}</div>
                <div class="stat-label">Source Papers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.year_range || '2011 - 2024'}</div>
                <div class="stat-label">Years Covered</div>
            </div>
        </div>
        ${data.subjects ? `
            <div style="margin-top:14px;">
                <h4 style="font-size:0.88rem; color:#ffffff; margin-bottom:10px;">Questions by Subject</h4>
                ${Object.entries(data.subjects).map(([sub, count]) => `
                    <div class="stat-bar-container">
                        <div class="stat-bar-label">
                            <span>${sub}</span>
                            <span>${count}</span>
                        </div>
                        <div class="stat-bar">
                            <div class="stat-bar-fill" style="width:${(count / 3000) * 100}%"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        ` : ''}
    `;
}

// ─── Utilities ──────────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'block';
    clearTimeout(window.__toastTimer);
    window.__toastTimer = setTimeout(() => {
        toast.style.display = 'none';
    }, 2500);
}
