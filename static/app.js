/**
 * RBI RAG Assistant - Frontend Application
 */

// ---- State ----
const state = {
    isLoading: false,
    queryHistory: JSON.parse(localStorage.getItem('rbiRagHistory') || '[]'),
};

// ---- DOM Elements ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const queryInput = $('#queryInput');
const searchBtn = $('#searchBtn');
const suggestionsGrid = $('#suggestionsGrid');
const resultsSection = $('#resultsSection');
const loadingState = $('#loadingState');
const answerCard = $('#answerCard');
const answerContent = $('#answerContent');
const answerMeta = $('#answerMeta');
const sourcesCard = $('#sourcesCard');
const sourcesList = $('#sourcesList');
const sourcesCount = $('#sourcesCount');
const historySection = $('#historySection');
const historyList = $('#historyList');
const heroSection = $('#heroSection');
const statusIndicator = $('#statusIndicator');
const settingsBtn = $('#settingsBtn');
const modalOverlay = $('#modalOverlay');
const modalClose = $('#modalClose');
const rebuildBtn = $('#rebuildBtn');
const scrapeBtn = $('#scrapeBtn');
const statsGrid = $('#statsGrid');

// ---- API ----
const API_BASE = '';

async function apiQuery(question, topK = 5) {
    const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK }),
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Query failed');
    }
    return response.json();
}

async function apiStats() {
    const response = await fetch(`${API_BASE}/api/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    return response.json();
}

async function apiSuggestions() {
    const response = await fetch(`${API_BASE}/api/suggestions`);
    if (!response.ok) throw new Error('Failed to fetch suggestions');
    return response.json();
}

async function apiHealth() {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
}

async function apiRebuild() {
    const response = await fetch(`${API_BASE}/api/rebuild`, { method: 'POST' });
    if (!response.ok) throw new Error('Rebuild failed');
    return response.json();
}

async function apiScrape(live = false) {
    const response = await fetch(`${API_BASE}/api/scrape`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ live }),
    });
    if (!response.ok) throw new Error('Scrape failed');
    return response.json();
}

// ---- Search Logic ----
async function performSearch(question) {
    if (!question.trim() || state.isLoading) return;

    state.isLoading = true;
    searchBtn.disabled = true;

    // Show results section and loading
    resultsSection.style.display = 'block';
    loadingState.style.display = 'block';
    answerCard.style.display = 'none';
    sourcesCard.style.display = 'none';

    // Collapse hero
    heroSection.style.display = 'none';
    suggestionsGrid.style.display = 'none';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const result = await apiQuery(question);

        // Render answer
        answerContent.innerHTML = formatAnswer(result.answer);
        const providerTag = result.provider ? ` • ${result.provider.toUpperCase()} (${result.model || ''})` : '';
        answerMeta.textContent = `Response time: ${result.response_time}s • ${result.sources.length} sources found${providerTag}`;

        // Render sources
        renderSources(result.sources);

        // Show cards
        loadingState.style.display = 'none';
        answerCard.style.display = 'block';
        sourcesCard.style.display = 'block';

        // Save to history
        addToHistory(question, result.response_time);

    } catch (error) {
        loadingState.style.display = 'none';
        answerCard.style.display = 'block';
        answerContent.innerHTML = `
            <div style="color: var(--error); display: flex; align-items: center; gap: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
                <strong>Error:</strong> ${escapeHtml(error.message)}
            </div>
            <p style="margin-top: 8px; color: var(--text-muted);">
                Please ensure the backend server is running and try again.
            </p>
        `;
        answerMeta.textContent = 'Error occurred';
    } finally {
        state.isLoading = false;
        searchBtn.disabled = false;
    }
}

// ---- Rendering ----
function formatAnswer(text) {
    // Convert markdown-like formatting to HTML
    let html = escapeHtml(text);

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

    // Headers: ### text
    html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*?)$/gm, '<h3>$1</h3>');

    // Italic: *text*
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Code: `text`
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Horizontal rules
    html = html.replace(/^---+$/gm, '<hr>');
    html = html.replace(/^\*\*\*+$/gm, '<hr>');

    // Bullet lists
    html = html.replace(/^[\-\•] (.*?)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>\n?)+/gs, (match) => `<ul>${match}</ul>`);

    // Numbered lists
    html = html.replace(/^\d+\.\s(.*?)$/gm, '<li>$1</li>');

    // Line breaks to paragraphs
    html = html.split('\n\n').map(p => {
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<h3>') || p.startsWith('<ul>') || p.startsWith('<ol>') || p.startsWith('<hr')) return p;
        return `<p>${p}</p>`;
    }).join('');

    // Clean up remaining single newlines within paragraphs
    html = html.replace(/\n/g, '<br>');

    return html;
}

function renderSources(sources) {
    sourcesCount.textContent = `${sources.length} source${sources.length !== 1 ? 's' : ''}`;

    sourcesList.innerHTML = sources.map(source => `
        <a class="source-item" href="${escapeHtml(source.url)}" target="_blank" rel="noopener">
            <div class="source-item-header">
                <span class="source-type-badge ${escapeHtml(source.type)}">${formatType(source.type)}</span>
                <span class="source-item-title">${escapeHtml(source.title)}</span>
            </div>
            <div class="source-item-snippet">${escapeHtml(source.snippet || '')}</div>
            <div class="source-item-meta">
                ${source.date ? `
                    <span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                            <line x1="16" y1="2" x2="16" y2="6"/>
                            <line x1="8" y1="2" x2="8" y2="6"/>
                            <line x1="3" y1="10" x2="21" y2="10"/>
                        </svg>
                        ${escapeHtml(source.date)}
                    </span>
                ` : ''}
                <span>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/>
                        <polyline points="15 3 21 3 21 9"/>
                        <line x1="10" y1="14" x2="21" y2="3"/>
                    </svg>
                    rbi.org.in
                </span>
            </div>
        </a>
    `).join('');
}

function formatType(type) {
    const typeMap = {
        'press_release': 'Press Release',
        'notification': 'Notification',
        'master_direction': 'Master Direction',
    };
    return typeMap[type] || type;
}

function renderSuggestions(suggestions) {
    suggestionsGrid.innerHTML = suggestions.slice(0, 6).map(s => `
        <button class="suggestion-chip" data-query="${escapeHtml(s)}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
            </svg>
            ${escapeHtml(s)}
        </button>
    `).join('');

    // Attach event listeners
    suggestionsGrid.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.dataset.query;
            autoResizeTextarea();
            performSearch(chip.dataset.query);
        });
    });
}

function renderHistory() {
    if (state.queryHistory.length === 0) {
        historySection.style.display = 'none';
        return;
    }

    historySection.style.display = 'block';
    historyList.innerHTML = state.queryHistory.slice(0, 5).map(item => `
        <div class="history-item" data-query="${escapeHtml(item.query)}">
            <div class="history-item-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                </svg>
            </div>
            <span class="history-item-text">${escapeHtml(item.query)}</span>
            <span class="history-item-time">${item.time}s</span>
        </div>
    `).join('');

    historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            queryInput.value = item.dataset.query;
            autoResizeTextarea();
            performSearch(item.dataset.query);
        });
    });
}

function addToHistory(query, responseTime) {
    state.queryHistory.unshift({
        query,
        time: responseTime,
        timestamp: Date.now(),
    });
    if (state.queryHistory.length > 20) state.queryHistory.pop();
    localStorage.setItem('rbiRagHistory', JSON.stringify(state.queryHistory));
    renderHistory();
}

// ---- Stats Modal ----
async function showStatsModal() {
    try {
        const stats = await apiStats();
        const health = await apiHealth();

        const activeProviderText = health.active_provider 
            ? `${health.active_provider === 'gemini' ? 'Google Gemini' : 'Ollama'} (${health.active_model || ''})`
            : 'None (Retrieval Only)';

        statsGrid.innerHTML = `
            <div class="stat-card">
                <div class="stat-label">Total Documents</div>
                <div class="stat-value">${stats.total_documents}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active AI Provider</div>
                <div class="stat-value small ${health.llm_available ? 'success' : 'warning'}">
                    ${activeProviderText}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Google GenAI</div>
                <div class="stat-value small ${health.gemini_available ? 'success' : 'muted'}">
                    ${health.gemini_available ? '● Ready' : '○ Not Configured'}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Ollama (Local)</div>
                <div class="stat-value small ${health.ollama_available ? 'success' : 'warning'}">
                    ${health.ollama_available ? '● Connected' : '○ Offline'}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Embedding Model</div>
                <div class="stat-value small">${stats.embedding_model}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vector Store</div>
                <div class="stat-value small ${stats.vectorstore_exists ? 'success' : 'error'}">
                    ${stats.vectorstore_exists ? '● Indexed' : '○ Not Found'}
                </div>
            </div>
            <div class="stat-card" style="grid-column: 1 / -1;">
                <div class="stat-label">Document Types</div>
                <div class="stat-value small">
                    ${Object.entries(stats.types || {}).map(([k, v]) => `${formatType(k)}: ${v}`).join(' &nbsp;•&nbsp; ')}
                </div>
            </div>
        `;
    } catch (error) {
        statsGrid.innerHTML = `
            <div class="stat-card" style="grid-column: 1 / -1;">
                <div class="stat-label">Error</div>
                <div class="stat-value small error">${escapeHtml(error.message)}</div>
            </div>
        `;
    }

    modalOverlay.style.display = 'flex';
}

// ---- Health Check ----
async function checkHealth() {
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');

    try {
        const health = await apiHealth();
        if (health.llm_available) {
            statusDot.className = 'status-dot active';
            if (health.active_provider === 'gemini') {
                statusText.textContent = `Gemini Ready (${health.active_model || 'flash'})`;
            } else if (health.active_provider === 'ollama') {
                statusText.textContent = `Ollama Ready (${health.active_model || 'llama3'})`;
            } else {
                statusText.textContent = 'AI Ready';
            }
        } else {
            statusDot.className = 'status-dot warning';
            statusText.textContent = 'Retrieval Only';
        }
    } catch {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Offline';
    }
}

// ---- Utility ----
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
}

// ---- Event Listeners ----
queryInput.addEventListener('input', autoResizeTextarea);

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        performSearch(queryInput.value);
    }
});

searchBtn.addEventListener('click', () => {
    performSearch(queryInput.value);
});

settingsBtn.addEventListener('click', showStatsModal);

modalClose.addEventListener('click', () => {
    modalOverlay.style.display = 'none';
});

modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) {
        modalOverlay.style.display = 'none';
    }
});

rebuildBtn.addEventListener('click', async () => {
    rebuildBtn.disabled = true;
    rebuildBtn.textContent = 'Rebuilding...';
    try {
        await apiRebuild();
        alert('Index rebuilt successfully!');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        rebuildBtn.disabled = false;
        rebuildBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"/>
                <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/>
            </svg>
            Rebuild Index
        `;
    }
});

scrapeBtn.addEventListener('click', async () => {
    if (!confirm('This will scrape live data from RBI website. Continue?')) return;
    scrapeBtn.disabled = true;
    scrapeBtn.textContent = 'Scraping...';
    try {
        await apiScrape(true);
        alert('Live data scraped and indexed successfully!');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        scrapeBtn.disabled = false;
        scrapeBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9"/>
            </svg>
            Scrape Live Data
        `;
    }
});

// ---- Initialize ----
async function init() {
    // Check health
    checkHealth();
    setInterval(checkHealth, 30000); // Check every 30s

    // Load suggestions
    try {
        const data = await apiSuggestions();
        renderSuggestions(data.suggestions);
    } catch {
        // Fallback suggestions
        renderSuggestions([
            "What is the current repo rate?",
            "Explain the Regulatory Sandbox framework",
            "What are digital lending guidelines?",
            "KYC norms for banks",
            "UPI transaction limits",
            "What is CBDC Digital Rupee?",
        ]);
    }

    // Render history
    renderHistory();

    // Focus input
    queryInput.focus();
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
