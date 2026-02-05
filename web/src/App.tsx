import { useEffect, useMemo, useState } from 'react'
import { api } from './api/client'
import type { EvidenceItem, RunCreateResponse, ScoreResponse, ExplanationResponse } from './types'

const defaultModels = ['baseline_v1', 'lr_v1', 'lr_v2']

function App() {
  const [claim, setClaim] = useState('')
  const [query, setQuery] = useState('')
  const [maxRecords, setMaxRecords] = useState(25)
  const [modelId, setModelId] = useState('baseline_v1')
  const [includeExplanation, setIncludeExplanation] = useState(false)
  const [run, setRun] = useState<RunCreateResponse | null>(null)
  const [score, setScore] = useState<ScoreResponse | null>(null)
  const [evidence, setEvidence] = useState<EvidenceItem[]>([])
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [chatQ, setChatQ] = useState('')
  const [chatA, setChatA] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [models, setModels] = useState<string[]>(defaultModels)

  const runId = run?.run_id || ''

  useEffect(() => {
    api.getModels().then(r => setModels(r.models)).catch(() => setModels(defaultModels))
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const id = params.get('run_id')
    if (id) {
      hydrateRun(id)
    }
  }, [])

  const canSubmit = useMemo(() => claim.trim().length > 0, [claim])

  async function hydrateRun(id: string) {
    setLoading(true)
    setError('')
    try {
      const [scoreResp, evidenceResp] = await Promise.all([
        api.getScore(id),
        api.getEvidence(id)
      ])
      setScore(scoreResp)
      setEvidence(evidenceResp)
      try {
        const expl = await api.getExplanation(id)
        setExplanation(expl)
      } catch {
        setExplanation(null)
      }
      setRun({
        run_id: id,
        status: 'completed',
        score: scoreResp.score,
        label: scoreResp.label,
        top_contributions: [],
        evidence_count: evidenceResp.length,
        features: [],
        explanation: null
      })
    } catch (e: any) {
      setError(e.message || 'Failed to load run')
    } finally {
      setLoading(false)
    }
  }

  async function onSubmit() {
    setLoading(true)
    setError('')
    setChatA('')
    try {
      const resp = await api.createRun({
        claim_text: claim,
        query_text: query || null,
        max_records: maxRecords,
        model_id: modelId,
        include_explanation: includeExplanation
      })
      setRun(resp)
      setScore(await api.getScore(resp.run_id))
      setEvidence(await api.getEvidence(resp.run_id))
      try {
        const expl = await api.getExplanation(resp.run_id)
        setExplanation(expl)
      } catch {
        setExplanation(null)
      }
      const params = new URLSearchParams(window.location.search)
      params.set('run_id', resp.run_id)
      window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`)
    } catch (e: any) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function onChat() {
    if (!runId || !chatQ.trim()) return
    setLoading(true)
    setError('')
    try {
      const resp = await api.chat(runId, chatQ)
      setChatA(resp.answer)
    } catch (e: any) {
      setError(e.message || 'Chat failed')
    } finally {
      setLoading(false)
    }
  }

  function copyLink() {
    if (!runId) return
    const url = `${window.location.origin}${window.location.pathname}?run_id=${runId}`
    navigator.clipboard.writeText(url)
  }

  return (
    <div className="page">
      <header className="header">
        <h1>TrustLens Demo</h1>
        <p>Submit a claim, score it, and inspect evidence + explanations.</p>
      </header>

      <section className="card">
        <h2>New Run</h2>
        <label>Claim</label>
        <textarea value={claim} onChange={(e) => setClaim(e.target.value)} rows={4} />

        <label>Query (optional)</label>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="If empty, claim is used" />

        <div className="row">
          <div>
            <label>Max Records</label>
            <input type="number" value={maxRecords} onChange={(e) => setMaxRecords(Number(e.target.value))} min={1} max={250} />
          </div>
          <div>
            <label>Model</label>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="Custom model id"
            />
          </div>
          <div className="checkbox">
            <label>
              <input type="checkbox" checked={includeExplanation} onChange={(e) => setIncludeExplanation(e.target.checked)} />
              Include explanation
            </label>
          </div>
        </div>

        <button disabled={!canSubmit || loading} onClick={onSubmit}>Run</button>
        {loading && <p className="muted">Loading…</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {run && (
        <section className="card">
          <div className="row space">
            <h2>Results</h2>
            <button onClick={copyLink} disabled={!runId}>Copy Share Link</button>
          </div>
          <p><strong>Run ID:</strong> {run.run_id}</p>
          <p><strong>Status:</strong> {run.status}</p>
          {score && (
            <p><strong>Score:</strong> {score.score.toFixed(3)} ({score.label})</p>
          )}

          {run.top_contributions?.length > 0 && (
            <div>
              <h3>Top Contributions</h3>
              <ul>
                {run.top_contributions.map((c) => (
                  <li key={c.feature_name}>{c.feature_name}: {c.contribution.toFixed(3)}</li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <h3>Evidence</h3>
            {evidence.length === 0 ? (
              <p className="muted">No evidence found.</p>
            ) : (
              <ul>
                {evidence.map((e) => (
                  <li key={e.url}>
                    <strong>{e.domain}</strong> — {e.title || 'Untitled'}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div>
            <h3>Explanation</h3>
            {explanation ? (
              <p>{explanation.response_text}</p>
            ) : (
              <p className="muted">No explanation available.</p>
            )}
          </div>
        </section>
      )}

      {run && (
        <section className="card">
          <h2>Chat</h2>
          <input value={chatQ} onChange={(e) => setChatQ(e.target.value)} placeholder="Ask about this run" />
          <button disabled={loading || !chatQ.trim()} onClick={onChat}>Ask</button>
          {chatA && <div className="chat"><strong>Answer:</strong> {chatA}</div>}
        </section>
      )}
    </div>
  )
}

export default App
