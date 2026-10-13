import { formatOutputString } from '../utils'

const ACTION_META = {
  CALL_PDF_AGENT: { label: 'PDF Specialist', badge: 'pdf' },
  CALL_SQL_AGENT: { label: 'SQL Specialist', badge: 'sql' },
}

export default function TraceLog({ trace }) {
  return (
    <div className="trace-steps-list">
      <h4>Trace Log</h4>
      <div className="steps-scroll">
        {trace.length === 0 ? (
          <div className="no-trace-yet">Ask a question to see real-time agent trace logs.</div>
        ) : (
          trace.map((step) => {
            const meta = ACTION_META[step.action] ?? { label: step.action, badge: 'finish' }
            return (
              <div className="trace-step-card" key={step.iteration}>
                <div className="trace-step-header">
                  <span>Iteration #{step.iteration}</span>
                  <span className={`trace-action-badge ${meta.badge}`}>{meta.label}</span>
                </div>
                <div className="trace-step-instructions">
                  <strong>Goal:</strong> {step.instructions}
                </div>
                {step.output && (
                  <div className="trace-step-output">
                    {formatOutputString(step.output, step.action)}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
