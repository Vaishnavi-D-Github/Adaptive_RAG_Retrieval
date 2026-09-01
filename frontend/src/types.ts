export type Role = "employee" | "hr";
export interface User { user_id: string; full_name: string; email: string; role: Role; is_active?: boolean }
export type Mode = "adaptive" | "fixed_3" | "fixed_5" | "fixed_10";
export interface Telemetry { [key: string]: any; initial_k?: number; predicted_k?: number; selected_k?: number; maximum_k?: number; retrieval_iterations?: number; retrieval_strategy?: string; predicted_k_confidence?: number; prediction_confidence?: number; k_escalated?: boolean; verification_performed?: boolean; verification_result?: unknown; num_retrieved_chunks?: number; num_chunks_used?: number; unique_documents?: number; total_tokens?: number; total_latency_ms?: number }
export interface Source { [key: string]: unknown; document?: string; source?: string; source_document?: string; page?: number | string; source_page?: number | string; rank?: number | string; distance?: number | string; text?: string; content?: string }
export interface QueryResult { mode: Mode; answer: string; sources: Source[]; telemetry: Telemetry; history_entry?: HistoryEntry }
export interface HistoryEntry { timestamp?: string; question?: string; mode?: Mode; answer?: string; sources?: Source[]; telemetry?: Telemetry; initial_k?: number; selected_k?: number; total_tokens?: number; total_latency_ms?: number }
