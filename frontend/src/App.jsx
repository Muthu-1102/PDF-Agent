import { useState, useRef } from "react";
import "./App.css";

/**
 * Agentic RAG — sidebar / chat / evidence-panel layout, re-themed to match
 * the provided reference UI (olive sidebar, cream chat area, mint evidence
 * panel). The retrieve -> evidence-check -> retry -> report trace rail and
 * evidence panel are still driven by the real API response.
 */

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const TRACE_STEPS = [
  { key: "retrieve", label: "Retrieve" },
  { key: "evidence", label: "Evidence" },
  { key: "report", label: "Report" },
];

const SUGGESTIONS = [
  { title: "Summarize the case file", sub: "across everything I've uploaded so far" },
  { title: "Find key facts", sub: "about the retrieval pipeline design" },
  { title: "Compare two sections", sub: "from different uploaded documents" },
  { title: "List open action items", sub: "mentioned anywhere in the record" },
  { title: "Explain a term", sub: "as it's defined in the case file" },
  { title: "Check a claim", sub: "against the evidence I've uploaded" },
];

export default function App() {
  // --- Case file (upload) state ---
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { text, error }
  const [caseFiles, setCaseFiles] = useState([]);
  const fileInputRef = useRef(null);
  const questionInputRef = useRef(null);

  // --- Conversation state ---
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(null); // 'retrieve' | 'evidence' | 'retry' | 'report'
  const [liveRetries, setLiveRetries] = useState(0);
  const [askedQuestion, setAskedQuestion] = useState("");
  const [result, setResult] = useState(null);

  async function submitFile(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus({ text: `Indexing ${file.name}…`, error: false });

    try {
      const res = await fetch("/upload/", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        setUploadStatus({ text: data.detail || "Upload failed.", error: true });
        return;
      }

      setCaseFiles((prev) => [
        { name: data.document_name, chunks: data.chunks_indexed, id: data.document_id },
        ...prev,
      ]);
      setUploadStatus({ text: `Indexed — ${data.chunks_indexed} chunks.`, error: false });
    } catch {
      setUploadStatus({ text: "Could not reach the server.", error: true });
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    submitFile(e.dataTransfer.files?.[0]);
  }

  function handleNewChat() {
    setQuestion("");
    setAskedQuestion("");
    setResult(null);
    setStage(null);
    setLiveRetries(0);
  }

  function applySuggestion(text) {
    setQuestion(text);
    questionInputRef.current?.focus();
  }

  async function handleAsk() {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setAskedQuestion(q);
    setQuestion("");
    setLoading(true);
    setResult(null);
    setLiveRetries(0);
    setStage("retrieve");

    const fetchPromise = fetch("/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    })
      .then((r) => r.json())
      .catch(() => null);

    await sleep(450);
    setStage("evidence");
    await sleep(500);

    const data = await fetchPromise;

    if (data && data.retries > 0) {
      for (let i = 0; i < data.retries; i++) {
        setStage("retry");
        setLiveRetries(i + 1);
        await sleep(380);
        setStage("evidence");
        await sleep(380);
      }
    }

    setStage("report");
    setResult(
      data || {
        answer: "Could not reach the server. Confirm the backend is running.",
        sources: [],
        confidence: 0,
        evidence_status: "insufficient",
        retries: 0,
      }
    );
    setLoading(false);
  }

  function nodeStatus(stepKey) {
    const order = ["retrieve", "evidence", "report"];
    if (!stage) return "pending";
    if (stage === "retry") {
      return stepKey === "evidence"
        ? "retry"
        : order.indexOf(stepKey) < order.indexOf("evidence")
        ? "done"
        : "pending";
    }
    const stageIndex = order.indexOf(stage);
    const stepIndex = order.indexOf(stepKey);
    if (stepIndex < stageIndex) return "done";
    if (stepIndex === stageIndex) return loading || stepKey !== "report" ? "active" : "done";
    return "pending";
  }

  const isInsufficient = result?.evidence_status === "insufficient";
  const avgConfidence = result?.confidence ?? null;

  return (
    <div className="aiva-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <p className="brand">
          RAG <span>AGENT</span>
        </p>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <span>+</span> New Chat
        </button>

        <div className="sidebar-section">
          <p className="sidebar-label">Case File</p>
          <div
            className={`sidebar-dropzone${isDragging ? " is-dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <p>Drop a document, or</p>
            <span>choose a file to index</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => submitFile(e.target.files?.[0])}
            />
          </div>

          {uploadStatus && (
            <p className={`sidebar-status${uploadStatus.error ? " is-error" : ""}`}>{uploadStatus.text}</p>
          )}

          {caseFiles.length > 0 && (
            <ul className="case-list">
              {caseFiles.map((f) => (
                <li className="case-item" key={f.id}>
                  <div className="case-item-name">{f.name}</div>
                  <div className="case-item-meta">{f.chunks} chunks indexed</div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="sidebar-footer">
          <span className="sidebar-chip">LangGraph</span>
          <span className="sidebar-chip">ChromaDB</span>
          <span className="sidebar-chip">Groq</span>
        </div>
      </aside>

      {/* Main column */}
      <div className="main-col">
        <header className="topbar">
          <div className="topbar-stats">
            <span className="stat-pill">
              Docs <b>{caseFiles.length}</b>
            </span>
            <span className="stat-pill">
              Confidence <b>{avgConfidence ?? "—"}</b>
            </span>
            <span className="stat-pill">
              Retries <b>{result?.retries ?? 0}</b>
            </span>
          </div>
          <div className="topbar-status">
            <span className="status-dot" />
            Backend online
          </div>
        </header>

        <main className="chat-area">
          {!askedQuestion ? (
            <div className="chat-hero">
              <h1 className="chat-watermark">
                RAG
                <small>Evidence-verified answers</small>
              </h1>

              <div className="suggestion-grid">
                {SUGGESTIONS.map((s) => (
                  <div className="suggestion-card" key={s.title} onClick={() => applySuggestion(s.title)}>
                    <p className="suggestion-title">{s.title}</p>
                    <p className="suggestion-sub">{s.sub}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              <div className="question-bubble">{askedQuestion}</div>

              <div className="trace-rail">
                {TRACE_STEPS.map((step) => {
                  const status = nodeStatus(step.key);
                  return (
                    <div className={`trace-node is-${status}`} key={step.key}>
                      <div className="trace-dot">
                        <span className="trace-line" />
                        {status === "done" ? "✓" : ""}
                      </div>
                      <span className="trace-label">{step.label}</span>
                      {step.key === "evidence" && liveRetries > 0 && (
                        <span className="trace-retry-badge">×{liveRetries}</span>
                      )}
                    </div>
                  );
                })}
              </div>

              {result && (
                <div className="report-card">
                  <div className="report-head">
                    <h3 className="report-title">Findings</h3>
                    {isInsufficient ? (
                      <span className="stamp is-low">Insufficient evidence</span>
                    ) : (
                      <span className="stamp">Confidence {result.confidence}</span>
                    )}
                  </div>
                  <p className="report-answer">{result.answer}</p>
                </div>
              )}
            </div>
          )}
        </main>

        <footer className="composer">
          <div className="composer-row">
            <input
              ref={questionInputRef}
              className="composer-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              placeholder="Send a message…"
            />
            <button className="send-btn" onClick={handleAsk} disabled={loading}>
              ➤
            </button>
          </div>
        </footer>
      </div>

      {/* Evidence panel */}
      <aside className="evidence-panel">
        <p className="evidence-panel-title">Evidence</p>
        {result?.sources?.length > 0 ? (
          <div className="evidence-list">
            {result.sources.map((s, i) => (
              <div className="evidence-tag" key={i}>
                <div className="evidence-tag-doc">{s.document}</div>
                <div className="evidence-tag-loc">
                  p.{s.page_number ?? "—"} · chunk {s.chunk_id}
                </div>
                <div className="evidence-bar-track">
                  <div className="evidence-bar-fill" style={{ width: `${Math.round(s.score * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="evidence-empty">
            Supporting <b>evidence</b> from your <b>case file</b> will appear here
          </p>
        )}
      </aside>
    </div>
  );
}