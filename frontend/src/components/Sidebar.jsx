const NAV_ITEMS = [
  { id: 'chat', icon: '💬', label: 'Chat Agent' },
  { id: 'sql', icon: '📊', label: 'SQL Analytics' },
  { id: 'pdf', icon: '📄', label: 'PDF Research' },
  { id: 'cache', icon: '💾', label: 'Semantic Cache' },
]

export default function Sidebar({ activeTab, onSelectTab }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-icon">🚀</div>
        <h2>FIN-RAG Dashboard</h2>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item${activeTab === item.id ? ' active' : ''}`}
            onClick={() => onSelectTab(item.id)}
          >
            <span className="icon">{item.icon}</span> {item.label}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="status-indicator">
          <span className="dot online"></span>
          <span>System Connected</span>
        </div>
      </div>
    </aside>
  )
}
