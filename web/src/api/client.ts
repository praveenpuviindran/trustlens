import type {
  RunCreateRequest,
  RunCreateResponse,
  ScoreResponse,
  EvidenceItem,
  ExplanationResponse,
  ChatResponse,
  ModelListResponse
} from '../types'

const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const API_BASE = RAW_BASE.endsWith('/api') ? RAW_BASE : `${RAW_BASE}/api`

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return (await res.json()) as T
}

export const api = {
  createRun(payload: RunCreateRequest): Promise<RunCreateResponse> {
    return request('/runs', { method: 'POST', body: JSON.stringify(payload) })
  },
  getScore(runId: string): Promise<ScoreResponse> {
    return request(`/runs/${runId}/score`)
  },
  getEvidence(runId: string): Promise<EvidenceItem[]> {
    return request(`/runs/${runId}/evidence`)
  },
  getExplanation(runId: string): Promise<ExplanationResponse> {
    return request(`/runs/${runId}/explanation`)
  },
  chat(runId: string, question: string): Promise<ChatResponse> {
    return request(`/runs/${runId}/chat`, { method: 'POST', body: JSON.stringify({ question }) })
  },
  async getModels(): Promise<ModelListResponse> {
    return request('/models')
  }
}
