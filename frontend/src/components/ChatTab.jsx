import { useRef, useState } from 'react'
import FlowVisualizer from './FlowVisualizer'
import TraceLog from './TraceLog'
import { sendChatMessage } from '../api'
import { formatText, sleep } from '../utils'

const INITIAL_MESSAGE = {
  sender: 'assistant',
  html: `<p>Welcome to FIN-RAG Master Supervisor. I can route your questions to specialized agents:</p>
    <ul>
      <li><strong>PDF Research Specialist:</strong> Searches unstructured document chunks (e.g., quarterly reports, Apple Q4 data).</li>
      <li><strong>SQL Analytics Specialist:</strong> Queries structured company & financial database tables.</li>
    </ul>
    <p>Ask me a financial question!</p>`,
}

export default function ChatTab({ onSqlResult, onPdfResult, onQuerySent, onCacheResult }) {
  const [messages, setMessages] = useState([INITIAL_MESSAGE])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [trace, setTrace] = useState([])
  const [activeNodes, setActiveNodes] = useState(new Set())
  const [activeConns, setActiveConns] = useState(new Set())
  const chatHistoryRef = useRef([])
  const messagesEndRef = useRef(null)

  function highlightNode(nodeKey) {
    setActiveNodes((prev) => new Set(prev).add(nodeKey))
    if (nodeKey === 'supervisor') setActiveConns((prev) => new Set(prev).add(0))
    if (nodeKey === 'pdf' || nodeKey === 'sql') setActiveConns((prev) => new Set(prev).add(1))
    if (nodeKey === 'finish') setActiveConns((prev) => new Set(prev).add(2))
  }

  function resetFlowAnimation() {
    setActiveNodes(new Set())
    setActiveConns(new Set())
    setTrace([])
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ block: 'end' })
    })
  }

  function appendMessage(sender, text, isCacheHit = false) {
    setMessages((prev) => [...prev, { sender, html: formatText(text), isCacheHit }])
    scrollToBottom()
  }

  async function animateTraceExecution(fullTrace) {
    if (!fullTrace || fullTrace.length === 0) return

    highlightNode('supervisor')
    await sleep(600)

    for (const step of fullTrace) {
      setTrace((prev) => [...prev, step])

      if (step.action === 'CALL_PDF_AGENT') {
        highlightNode('pdf')
        setActiveNodes((prev) => {
          const next = new Set(prev)
          next.delete('sql')
          return next
        })
        if (step.output) onPdfResult(step.output)
        await sleep(1000)
      } else if (step.action === 'CALL_SQL_AGENT') {
        highlightNode('sql')
        setActiveNodes((prev) => {
          const next = new Set(prev)
          next.delete('pdf')
          return next
        })
        if (step.output) onSqlResult(step.instructions, step.output)
        await sleep(1000)
      } else if (step.action === 'FINISH') {
        highlightNode('finish')
        await sleep(600)
      }
    }
  }

  async function handleSendMessage() {
    const query = input.trim()
    if (!query) return

    setInput('')
    appendMessage('user', query)
    onQuerySent()

    resetFlowAnimation()
    highlightNode('user')
    setIsTyping(true)

    try {
      const data = await sendChatMessage(query, chatHistoryRef.current)
      setIsTyping(false)

      if (data.cache_hit) {
        onCacheResult(query, true)
        highlightNode('supervisor')
        highlightNode('finish')
        appendMessage('assistant', data.answer, true)
      } else {
        onCacheResult(query, false)
        await animateTraceExecution(data.trace)
        appendMessage('assistant', data.answer, false)
      }

      chatHistoryRef.current = [
        ...chatHistoryRef.current,
        { role: 'user', content: query },
        { role: 'assistant', content: data.answer },
      ]
    } catch (error) {
      setIsTyping(false)
      appendMessage('assistant', `⚠️ Sorry, there was an error processing your request: ${error.message}`)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleSendMessage()
  }

  return (
    <div className="tab-content active" id="chat-tab">
      <div className="chat-layout">
        <div className="chat-container glass">
          <div className="chat-messages" id="chat-messages">
            {messages.map((msg, i) => (
              <div className={`message ${msg.sender}`} key={i}>
                <div className="avatar">{msg.sender === 'user' ? '👤' : '🤖'}</div>
                <div className="message-content">
                  {msg.isCacheHit && (
                    <div className="cache-hit-badge">⚡ Semantic Cache Hit (0ms)</div>
                  )}
                  <div dangerouslySetInnerHTML={{ __html: msg.html }} />
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="message assistant">
                <div className="avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              placeholder="Type a financial question..."
              autoComplete="off"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button className="send-btn" onClick={handleSendMessage}>
              <span>Send</span>
              <span className="arrow">➔</span>
            </button>
          </div>
        </div>

        <div className="trace-container glass">
          <h3>Agent Execution Flow</h3>
          <FlowVisualizer activeNodes={activeNodes} activeConns={activeConns} />
          <TraceLog trace={trace} />
        </div>
      </div>
    </div>
  )
}
