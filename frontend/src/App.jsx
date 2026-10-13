import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatTab from './components/ChatTab'
import SqlTab from './components/SqlTab'
import PdfTab from './components/PdfTab'
import CacheTab from './components/CacheTab'

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [sqlLog, setSqlLog] = useState({ instruction: null, output: null })
  const [pdfOutput, setPdfOutput] = useState(null)
  const [cacheStats, setCacheStats] = useState({ total: 0, hits: 0, log: [] })

  function handleSqlResult(instruction, output) {
    setSqlLog({ instruction, output })
  }

  function handlePdfResult(output) {
    setPdfOutput(output)
  }

  function handleQuerySent() {
    setCacheStats((prev) => ({ ...prev, total: prev.total + 1 }))
  }

  function handleCacheResult(query, isHit) {
    setCacheStats((prev) => ({
      total: prev.total,
      hits: isHit ? prev.hits + 1 : prev.hits,
      log: [{ query, isHit, timestamp: new Date().toLocaleTimeString() }, ...prev.log],
    }))
  }

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />
      <main className="main-content">
        <header className="main-header">
          <h1>FIN-RAG Master Supervisor</h1>
          <p>Multi-agent Routing &amp; Retrieval Pipeline</p>
        </header>

        {activeTab === 'chat' && (
          <ChatTab
            onSqlResult={handleSqlResult}
            onPdfResult={handlePdfResult}
            onQuerySent={handleQuerySent}
            onCacheResult={handleCacheResult}
          />
        )}
        {activeTab === 'sql' && <SqlTab sqlLog={sqlLog} />}
        {activeTab === 'pdf' && <PdfTab pdfOutput={pdfOutput} />}
        {activeTab === 'cache' && <CacheTab cacheStats={cacheStats} />}
      </main>
    </div>
  )
}
