export default function FlowVisualizer({ activeNodes, activeConns }) {
  const isActive = (key) => (activeNodes.has(key) ? ' active' : '')
  const connActive = (index) => (activeConns.has(index) ? ' active' : '')

  return (
    <div className="flow-visualizer">
      <div className={`flow-node user-node${isActive('user')}`}>
        <div className="node-icon">👤</div>
        <span>User</span>
      </div>
      <div className={`flow-connector${connActive(0)}`}></div>
      <div className={`flow-node supervisor-node${isActive('supervisor')}`}>
        <div className="node-icon">⚖️</div>
        <span>Supervisor</span>
      </div>
      <div className={`flow-connector branches${connActive(1)}`}>
        <div className="branch-line up"></div>
        <div className="branch-line down"></div>
      </div>
      <div className="flow-branches-container">
        <div className={`flow-node specialist-node pdf-branch${isActive('pdf')}`}>
          <div className="node-icon">📄</div>
          <span>PDF Specialist</span>
        </div>
        <div className={`flow-node specialist-node sql-branch${isActive('sql')}`}>
          <div className="node-icon">🗄️</div>
          <span>SQL Specialist</span>
        </div>
      </div>
      <div className={`flow-connector branches merge${connActive(2)}`}>
        <div className="branch-line up"></div>
        <div className="branch-line down"></div>
      </div>
      <div className={`flow-node finish-node${isActive('finish')}`}>
        <div className="node-icon">🏁</div>
        <span>Answer Synthesizer</span>
      </div>
    </div>
  )
}
