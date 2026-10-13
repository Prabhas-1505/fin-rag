import { parsePdfChunks, formatText } from '../utils'

export default function PdfTab({ pdfOutput }) {
  const chunks = parsePdfChunks(pdfOutput)

  return (
    <div className="tab-content active" id="pdf-tab">
      <div className="pane glass">
        <div className="pane-header">
          <h3>PDF Retrieval Chunks</h3>
          <span className="metric">{chunks.length} chunks retrieved</span>
        </div>
        <div className="pdf-chunks-grid">
          {chunks.length === 0 ? (
            <div className="empty-state">
              <span className="empty-icon">📄</span>
              <p>Ask a question about financial documents or reports to retrieve relevant document chunks.</p>
            </div>
          ) : (
            chunks.map((chunk, i) => (
              <div className="pdf-chunk-card" key={i}>
                <div className="pdf-chunk-meta">Source: {chunk.source}</div>
                <div
                  className="pdf-chunk-text"
                  dangerouslySetInnerHTML={{ __html: formatText(chunk.content) }}
                />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
