import type { HistoryEntry, Mode, QueryResult, Role, User } from "../types";

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  try {
    const response = await fetch(path, { credentials: "include", headers: { "Content-Type": "application/json", ...init.headers }, ...init });
    const payload = await response.json().catch(() => ({ error: "Invalid server response." }));
    if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
    return payload as T;
  } catch (error) { if (error instanceof TypeError) throw new Error("Unable to connect to the RAG server."); throw error; }
}
export const api = {
  me: () => request<{ user: User }>("/api/me"),
  login: (email: string, password: string) => request<{ user: User }>("/api/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  signup: (data: { full_name: string; email: string; password: string; confirm_password: string; role: Role }) => request<{ message: string }>("/api/signup", { method: "POST", body: JSON.stringify(data) }),
  logout: () => request("/api/logout", { method: "POST", body: "{}" }),
  query: (query: string, mode: Mode) => request<QueryResult>("/api/query", { method: "POST", body: JSON.stringify({ query, mode }) }),
  history: () => request<{ history: HistoryEntry[]; rag_runs?: HistoryEntry[]; schema_note?: string }>("/api/history"),
  upload: (filename: string, content_base64: string) => request<{ filename: string; chunk_count: number; message: string }>("/api/hr/upload", { method: "POST", body: JSON.stringify({ filename, content_base64 }) })
};
