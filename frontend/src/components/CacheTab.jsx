export default function CacheTab({ cacheStats }) {
  const { total, hits, log } = cacheStats
  const efficiency = total === 0 ? 'N/A' : `${((hits / total) * 100).toFixed(1)}%`

  return (
    <div className="tab-content active" id="cache-tab">
      <div className="pane glass">
        <div className="pane-header">
          <h3>Semantic Cache Analytics</h3>
        </div>
        <div className="cache-metrics">
          <div className="metric-card glass-card">
            <span className="metric-label">Cache Efficiency</span>
            <span className="metric-val">{efficiency}</span>
          </div>
          <div className="metric-card glass-card">
            <span className="metric-label">Total Queries</span>
            <span className="metric-val">{total}</span>
          </div>
          <div className="metric-card glass-card">
            <span className="metric-label">Cache Hits</span>
            <span className="metric-val">{hits}</span>
          </div>
        </div>
        <div className="cache-logs-container">
          <h4>Cache Activity Log</h4>
          <div className="cache-log-list">
            {log.length === 0 ? (
              <div className="empty-state">No cache activity recorded yet.</div>
            ) : (
              log.map((entry, i) => (
                <div className="cache-log-item" key={i}>
                  <div>
                    <span
                      style={{
                        color: entry.isHit ? 'var(--success)' : 'var(--warning)',
                        fontWeight: 'bold',
                        marginRight: '8px',
                      }}
                    >
                      {entry.isHit ? 'HIT' : 'MISS'}
                    </span>
                    <span className="cache-log-query">
                      "{entry.query.length > 50 ? `${entry.query.substring(0, 50)}...` : entry.query}"
                    </span>
                  </div>
                  <span className="cache-log-time">{entry.timestamp}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
