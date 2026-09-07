import { ChangeEvent, DragEvent, useEffect, useState } from "react";
import { CheckCircle2, FileUp, LoaderCircle, UploadCloud } from "lucide-react";
import { api } from "../services/api";
import type { DocumentRecord } from "../types";

const MAX = 20 * 1024 * 1024;
const types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];

export function DocumentsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [accessLevel, setAccessLevel] = useState<"student" | "teacher" | "office">("student");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);

  const refresh = () => api.documents().then(data => setDocuments(data.documents || [])).catch(() => setDocuments([]));
  useEffect(() => { void refresh(); }, []);

  const choose = (candidate?: File) => {
    setError("");
    setSuccess("");
    if (!candidate) return;
    if (!types.includes(candidate.type) && !/\.(pdf|docx|txt)$/i.test(candidate.name)) return setError("Only PDF, DOCX and TXT files are supported.");
    if (candidate.size > MAX) return setError("Maximum upload size is 20 MB.");
    setFile(candidate);
  };

  const upload = async () => {
    if (!file) return setError("Choose a document first.");
    setBusy(true);
    setError("");
    try {
      const encoded = await readFile(file);
      const response = await api.upload(file.name, encoded, accessLevel);
      setSuccess(`${response.message} ${response.chunk_count} chunks indexed.`);
      setFile(null);
      setAccessLevel("student");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to upload document.");
    } finally {
      setBusy(false);
    }
  };

  return <div className="content-page">
    <div className="page-intro">
      <span className="eyebrow">HR DOCUMENT MANAGEMENT</span>
      <h1>Documents</h1>
      <p>Upload enterprise content. Original files and extracted chunks are stored continuously, then indexed for retrieval.</p>
    </div>
    <label className={`dropzone ${file ? "has-file" : ""}`} onDragOver={e => e.preventDefault()} onDrop={(e: DragEvent) => { e.preventDefault(); choose(e.dataTransfer.files[0]); }}>
      <input type="file" accept=".pdf,.docx,.txt" onChange={(e: ChangeEvent<HTMLInputElement>) => choose(e.target.files?.[0])}/>
      {file ? <><FileUp size={30}/><b>{file.name}</b><span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready to index</span></> : <><UploadCloud size={32}/><b>Drag and drop your document here</b><span>PDF, DOCX or TXT · Maximum 20 MB</span><em>Browse files</em></>}
    </label>
    <div className="form-field">
      <label htmlFor="access-level">Document access level</label>
        <select
          id="access-level"
          value={accessLevel}
          onChange={e =>
            setAccessLevel(
              e.target.value as "student" | "teacher" | "office"
            )
          }
        >
          <option value="student">Student — Student, Teacher and Office</option>
          <option value="teacher">Teacher — Teacher and Office</option>
          <option value="office">Office — Office only</option>
        </select>
        <small>
          Choose the lowest role that should be allowed to access this document.
        </small>
    </div>
    <button className="primary-button upload-button" onClick={upload} disabled={!file || busy}>{busy ? <><LoaderCircle className="spin"/> Uploading and indexing…</> : <><FileUp size={18}/> Upload and index document</>}</button>
    {error && <div className="form-error">{error}</div>}
    {success && <div className="upload-success"><CheckCircle2/>{success}</div>}
    <div className="history-list document-list">
      {documents.length ? documents.map(item => (
        <div className="history-item" key={item.document_id}>
          <div className="history-icon"><FileUp size={17}/></div>
          <div>
            <b>{item.original_filename || item.filename || "Uploaded document"}</b>
            <span>{item.status || "UNKNOWN"} · {item.chunk_count ?? 0} chunks · {item.embedding_count ?? 0} embeddings · {item.chroma_verified_count ?? 0} verified</span>
            <small>{item.document_id}</small>
          </div>
          <time>{item.created_at ? new Date(item.created_at).toLocaleString() : ""}</time>
        </div>
      )) : <div className="empty-list">No stored documents yet.</div>}
    </div>
  </div>;
}

function readFile(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
