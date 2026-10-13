function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

export function formatText(text) {
  return escapeHtml(text)
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(
      /`(.*?)`/g,
      '<code style="font-family: var(--font-mono); background-color: rgba(255,255,255,0.05); padding: 2px 4px; border-radius: 4px;">$1</code>'
    )
}

export function formatOutputString(output, action) {
  if (action === 'CALL_SQL_AGENT') {
    try {
      return JSON.stringify(JSON.parse(output), null, 2)
    } catch {
      return output
    }
  }
  return output
}

export function parseSqlOutput(output) {
  try {
    const rows = JSON.parse(output)
    if (!Array.isArray(rows) || rows.length === 0) {
      return { rows: [], columns: [], error: null, empty: true }
    }
    return { rows, columns: Object.keys(rows[0]), error: null, empty: false }
  } catch (e) {
    return { rows: [], columns: [], error: e.message, empty: false, raw: output }
  }
}

export function parsePdfChunks(output) {
  if (!output) return []
  const chunks = output.split('\n---\n')
  if (chunks.length === 0 || !chunks[0].trim()) return []

  return chunks.map((chunk, index) => {
    const sourceMatch = chunk.match(/Source:\s*(.*)/i)
    const contentMatch = chunk.match(/Content:\s*(.*)/is)
    return {
      source: sourceMatch ? sourceMatch[1] : `Reference #${index + 1}`,
      content: contentMatch ? contentMatch[1] : chunk,
    }
  })
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
