import { parseSqlOutput } from '../utils'

export default function SqlTab({ sqlLog }) {
  const { instruction, output } = sqlLog
  const parsed = output ? parseSqlOutput(output) : null

  let rowsCountLabel = '0 rows returned'
  let tableBody = (
    <tr>
      <td>Ask a question that queries metrics or company profiles to see SQL results.</td>
    </tr>
  )
  let tableHead = (
    <tr>
      <th>No Data</th>
    </tr>
  )

  if (parsed) {
    if (parsed.error) {
      rowsCountLabel = 'Error'
      tableHead = (
        <tr>
          <th>Result</th>
        </tr>
      )
      tableBody = (
        <tr>
          <td>
            Error parsing SQL: {parsed.error}
            <br />
            Raw Output: {parsed.raw}
          </td>
        </tr>
      )
    } else if (parsed.empty) {
      rowsCountLabel = '0 rows returned'
      tableHead = (
        <tr>
          <th>Result</th>
        </tr>
      )
      tableBody = (
        <tr>
          <td>Query executed, but returned 0 rows.</td>
        </tr>
      )
    } else {
      rowsCountLabel = `${parsed.rows.length} rows returned`
      tableHead = (
        <tr>
          {parsed.columns.map((col) => (
            <th key={col}>{col}</th>
          ))}
        </tr>
      )
      tableBody = parsed.rows.map((row, i) => (
        <tr key={i}>
          {parsed.columns.map((col) => (
            <td key={col}>{String(row[col])}</td>
          ))}
        </tr>
      ))
    }
  }

  return (
    <div className="tab-content active" id="sql-tab">
      <div className="pane glass">
        <div className="pane-header">
          <h3>SQL Database Results</h3>
          <span className="metric">{rowsCountLabel}</span>
        </div>
        <div className="sql-query-display">
          {instruction ? `-- Inferred Tool Instruction:\n${instruction}` : '-- No SQL executed yet. Ask a database question in Chat.'}
        </div>
        <div className="table-scroll-container">
          <table className="data-table">
            <thead>{tableHead}</thead>
            <tbody>{tableBody}</tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
