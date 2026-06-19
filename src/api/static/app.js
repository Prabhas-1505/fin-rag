document.addEventListener('DOMContentLoaded', () => {
    console.log("[FIN-RAG UI] app.js loaded successfully!");
    // Navigation & Tabs
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Chat Logic
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const traceStepsList = document.getElementById('trace-steps-list');

    // UI Nodes for Visual Flow
    const nodes = {
        user: document.getElementById('node-user'),
        supervisor: document.getElementById('node-supervisor'),
        pdf: document.getElementById('node-pdf'),
        sql: document.getElementById('node-sql'),
        finish: document.getElementById('node-finish'),
        conns: [
            document.getElementById('conn-1'),
            document.getElementById('conn-2'),
            document.getElementById('conn-3')
        ]
    };

    // Cache Stats
    let totalQueries = 0;
    let cacheHits = 0;
    const cacheEfficiencyEl = document.getElementById('cache-efficiency');
    const totalQueriesEl = document.getElementById('cache-total-queries');
    const cacheHitsEl = document.getElementById('cache-hits-count');
    const cacheLogList = document.getElementById('cache-log-list');

    // Session Chat History
    let chatHistory = [];

    // SQL Tab Elements
    const sqlQueryDisplay = document.getElementById('sql-query-display');
    const sqlDataTable = document.getElementById('sql-data-table');
    const sqlRowsCount = document.getElementById('sql-rows-count');

    // PDF Tab Elements
    const pdfChunksGrid = document.getElementById('pdf-chunks-grid');
    const pdfChunksCount = document.getElementById('pdf-chunks-count');

    // Listeners
    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleSendMessage();
    });

    async function handleSendMessage() {
        const query = chatInput.value.trim();
        console.log("[FIN-RAG UI] Send message triggered. Query:", query);
        if (!query) return;

        // Clear input
        chatInput.value = '';

        // Add user message to UI
        appendMessage('user', query);
        totalQueries++;
        updateCacheStats();

        // Visualizer starting state
        resetFlowAnimation();
        highlightNode('user');
        
        // Show Typing Indicator
        const typingIndicator = showTypingIndicator();

        try {
            // API Call
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: query,
                    history: chatHistory
                })
            });

            if (!response.ok) throw new Error("API call failed");

            const data = await response.json();

            // Remove typing indicator
            typingIndicator.remove();

            // Handle Cache Status & Output
            if (data.cache_hit) {
                cacheHits++;
                updateCacheStats();
                logCacheActivity(query, true);

                // UI visual jump to finish
                highlightNode('supervisor');
                highlightNode('finish');
                
                appendMessage('assistant', data.answer, true);
            } else {
                logCacheActivity(query, false);
                // Animate trace and populate specialist views
                await animateTraceExecution(data.trace);
                appendMessage('assistant', data.answer, false);
            }

            // Save to chat history
            chatHistory.push({ role: 'user', content: query });
            chatHistory.push({ role: 'assistant', content: data.answer });

        } catch (error) {
            console.error("Error sending message:", error);
            typingIndicator.remove();
            appendMessage('assistant', `⚠️ Sorry, there was an error processing your request: ${error.message}`);
        }
    }

    function appendMessage(sender, text, isCacheHit = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        let cacheBadgeHtml = isCacheHit ? `<div class="cache-hit-badge">⚡ Semantic Cache Hit (0ms)</div>` : '';
        
        messageDiv.innerHTML = `
            <div class="avatar">${sender === 'user' ? '👤' : '🤖'}</div>
            <div class="message-content">
                ${cacheBadgeHtml}
                <p>${formatText(text)}</p>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.classList.add('message', 'assistant');
        indicatorDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(indicatorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return indicatorDiv;
    }

    function formatText(text) {
        // Simple markdown parsing for bullets and strong tags
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code style="font-family: var(--font-mono); background-color: rgba(255,255,255,0.05); padding: 2px 4px; border-radius: 4px;">$1</code>');
    }

    function resetFlowAnimation() {
        Object.values(nodes).forEach(node => {
            if (node && node.classList) node.classList.remove('active');
        });
        nodes.conns.forEach(conn => {
            if (conn) conn.classList.remove('active');
        });
        traceStepsList.innerHTML = '';
    }

    function highlightNode(nodeKey) {
        const node = nodes[nodeKey];
        if (node) {
            node.classList.add('active');
            // Animate connectors leading to it
            if (nodeKey === 'supervisor') nodes.conns[0].classList.add('active');
            if (nodeKey === 'pdf' || nodeKey === 'sql') nodes.conns[1].classList.add('active');
            if (nodeKey === 'finish') nodes.conns[2].classList.add('active');
        }
    }

    async function animateTraceExecution(trace) {
        if (!trace || trace.length === 0) return;

        highlightNode('supervisor');
        await sleep(600);

        for (const step of trace) {
            // Render Trace Card
            renderTraceCard(step);

            if (step.action === 'CALL_PDF_AGENT') {
                highlightNode('pdf');
                nodes.sql.classList.remove('active');
                if (step.output) parseAndRenderPDFOutput(step.output);
                await sleep(1000);
            } else if (step.action === 'CALL_SQL_AGENT') {
                highlightNode('sql');
                nodes.pdf.classList.remove('active');
                if (step.output) parseAndRenderSQLOutput(step.instructions, step.output);
                await sleep(1000);
            } else if (step.action === 'FINISH') {
                highlightNode('finish');
                await sleep(600);
            }
        }
    }

    function renderTraceCard(step) {
        // Clear empty state
        const emptyState = traceStepsList.querySelector('.no-trace-yet');
        if (emptyState) emptyState.remove();

        const card = document.createElement('div');
        card.classList.add('trace-step-card');
        
        let actionLabel = step.action;
        let badgeClass = 'finish';
        if (step.action === 'CALL_PDF_AGENT') { actionLabel = 'PDF Specialist'; badgeClass = 'pdf'; }
        if (step.action === 'CALL_SQL_AGENT') { actionLabel = 'SQL Specialist'; badgeClass = 'sql'; }

        card.innerHTML = `
            <div class="trace-step-header">
                <span>Iteration #${step.iteration}</span>
                <span class="trace-action-badge ${badgeClass}">${actionLabel}</span>
            </div>
            <div class="trace-step-instructions"><strong>Goal:</strong> ${step.instructions}</div>
            ${step.output ? `<div class="trace-step-output">${formatOutputString(step.output, step.action)}</div>` : ''}
        `;
        traceStepsList.appendChild(card);
        traceStepsList.scrollTop = traceStepsList.scrollHeight;
    }

    function formatOutputString(output, action) {
        if (action === 'CALL_SQL_AGENT') {
            try {
                // Prettify JSON output for SQL
                const parsed = JSON.parse(output);
                return JSON.stringify(parsed, null, 2);
            } catch (e) {
                return output;
            }
        }
        return output;
    }

    function parseAndRenderSQLOutput(query, output) {
        sqlQueryDisplay.textContent = `-- Inferred Tool Instruction:\n${query}`;
        
        try {
            const rows = JSON.parse(output);
            if (!Array.isArray(rows) || rows.length === 0) {
                showEmptySQLTable("Query executed, but returned 0 rows.");
                sqlRowsCount.textContent = "0 rows returned";
                return;
            }

            sqlRowsCount.textContent = `${rows.length} rows returned`;

            // Extract columns
            const columns = Object.keys(rows[0]);
            
            // Build headers
            let headerHtml = '<tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
            
            // Build rows
            let bodyHtml = rows.map(row => {
                return '<tr>' + columns.map(col => `<td>${row[col]}</td>`).join('') + '</tr>';
            }).join('');

            sqlDataTable.innerHTML = `
                <thead>${headerHtml}</thead>
                <tbody>${bodyHtml}</tbody>
            `;

        } catch (e) {
            console.error("Error parsing SQL output:", e);
            showEmptySQLTable(`Error parsing SQL: ${e.message}\nRaw Output: ${output}`);
            sqlRowsCount.textContent = "Error";
        }
    }

    function showEmptySQLTable(message) {
        sqlDataTable.innerHTML = `
            <thead><tr><th>Result</th></tr></thead>
            <tbody><tr><td>${message}</td></tr></tbody>
        `;
    }

    function parseAndRenderPDFOutput(output) {
        pdfChunksGrid.innerHTML = '';
        
        // Split by standard LC document delimiter
        const chunks = output.split('\n---\n');
        
        if (chunks.length === 0 || !chunks[0].trim()) {
            pdfChunksCount.textContent = '0 chunks retrieved';
            pdfChunksGrid.innerHTML = `
                <div class="empty-state">
                    <p>No document chunks found.</p>
                </div>
            `;
            return;
        }

        pdfChunksCount.textContent = `${chunks.length} chunks retrieved`;

        chunks.forEach((chunk, index) => {
            // Extract Source and Content
            const sourceMatch = chunk.match(/Source:\s*(.*)/i);
            const contentMatch = chunk.match(/Content:\s*(.*)/is);
            
            const source = sourceMatch ? sourceMatch[1] : `Reference #${index + 1}`;
            const content = contentMatch ? contentMatch[1] : chunk;

            const chunkCard = document.createElement('div');
            chunkCard.classList.add('pdf-chunk-card');
            chunkCard.innerHTML = `
                <div class="pdf-chunk-meta">Source: ${source}</div>
                <div class="pdf-chunk-text">${formatText(content)}</div>
            `;
            pdfChunksGrid.appendChild(chunkCard);
        });
    }

    // Cache Stats Helpers
    function updateCacheStats() {
        totalQueriesEl.textContent = totalQueries;
        cacheHitsEl.textContent = cacheHits;
        
        if (totalQueries === 0) {
            cacheEfficiencyEl.textContent = 'N/A';
        } else {
            const efficiency = (cacheHits / totalQueries) * 100;
            cacheEfficiencyEl.textContent = `${efficiency.toFixed(1)}%`;
        }
    }

    function logCacheActivity(query, isHit) {
        const emptyState = cacheLogList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const logItem = document.createElement('div');
        logItem.classList.add('cache-log-item');
        
        const timestamp = new Date().toLocaleTimeString();

        logItem.innerHTML = `
            <div>
                <span style="color: ${isHit ? 'var(--success)' : 'var(--warning)'}; font-weight: bold; margin-right: 8px;">
                    ${isHit ? 'HIT' : 'MISS'}
                </span>
                <span class="cache-log-query">"${query.length > 50 ? query.substring(0, 50) + '...' : query}"</span>
            </div>
            <span class="cache-log-time">${timestamp}</span>
        `;
        cacheLogList.insertBefore(logItem, cacheLogList.firstChild);
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
