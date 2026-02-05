export type RunCreateRequest = {
  claim_text: string
  query_text?: string | null
  max_records: number
  model_id: string
  include_explanation: boolean
}

export type RunCreateResponse = {
  run_id: string
  status: string
  score: number
  label: string
  top_contributions: { feature_name: string; value: number; contribution: number; weight?: number }[]
  evidence_count: number
  features: { feature_group: string; feature_name: string; feature_value: number }[]
  explanation?: { summary: string; bullets: string[] } | null
}

export type ScoreResponse = {
  run_id: string
  model_version: string
  score: number
  label: string
  contributions?: Record<string, unknown> | null
}

export type EvidenceItem = {
  domain: string
  title?: string | null
  snippet?: string | null
  url: string
  published_at?: string | null
  retrieved_at: string
}

export type ExplanationResponse = {
  run_id: string
  model_id: string
  mode: string
  response_text: string
}

export type ChatResponse = {
  run_id: string
  answer: string
}

export type ModelListResponse = {
  models: string[]
}
