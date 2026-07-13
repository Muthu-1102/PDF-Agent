import { useState, useRef } from "react";
import "./App.css";

/**
 * Agentic RAG — "The Case File"
 *
 * The UI's one job: make the retrieve -> evidence-check -> retry -> report
 * loop from the LangGraph backend visible, not hide it behind a spinner.
 * Documents are treated as evidence, the Evidence Agent as a fact-checker,
 * and the trace rail below is driven by the real `retries` /
 * `evidence_status` values the API returns.
 */

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const TRACE_STEPS = [
  { key: "retrieve", label: "Retrieve" },
  { key: "evidence", label: "Weigh evidence" },
  { key: "report", label: "Report" },
];

export default function App() {
  // --- Case file (upload) state ---
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { text, error }
  const [caseFiles, setCaseFiles] = useState([]);
  const fileInputRef = useRef(null);

  // --- Interrogation (query) state ---
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(null); // 'retrieve' | 'evidence' | 'retry' | 'report'
  const [liveRetries, setLiveRetries] = useState(0);
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
        {
          name: data.document_name,
          chunks: data.chunks_indexed,
          id: data.document_id,
        },
        ...prev,
      ]);
      setUploadStatus({ text: `Indexed — ${data.chunks_indexed} chunks added to the record.`, error: false });
    } catch {
      setUploadStatus({ text: "Could not reach the server.", error: true });
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    submitFile(e.dataTransfer.files?.[0]);
  }

  async function handleAsk() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setLiveRetries(0);
    setStage("retrieve");

    const fetchPromise = fetch("/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })
      .then((r) => r.json())
      .catch(() => null);

    // Choreograph the trace rail so the retrieve -> evidence loop is
    // actually visible, then reconcile with the real response.
    await sleep(500);
    setStage("evidence");
    await sleep(550);

    const data = await fetchPromise;

    if (data && data.retries > 0) {
      for (let i = 0; i < data.retries; i++) {
        setStage("retry");
        setLiveRetries(i + 1);
        await sleep(420);
        setStage("evidence");
        await sleep(420);
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
      return stepKey === "evidence" ? "retry" : order.indexOf(stepKey) < order.indexOf("evidence") ? "done" : "pending";
    }
    const stageIndex = order.indexOf(stage);
    const stepIndex = order.indexOf(stepKey);
    if (stepIndex < stageIndex) return "done";
    if (stepIndex === stageIndex) return loading || stepKey !== "report" ? "active" : "done";
    return "pending";
  }

  const isInsufficient = result?.evidence_status === "insufficient";

  return (
    <div className="app-shell">
      <div className="container">
        {/* Top bar */}
        <div className="topbar">
          <div className="topbar-mark">
            <span className="glyph">§</span>
            <span className="name">Agentic RAG · Case File</span>
          </div>
          <div className="topbar-stack">
            <span className="stack-chip">LangGraph</span>
            <span className="stack-chip">ChromaDB</span>
            <span className="stack-chip">Groq</span>
          </div>
        </div>

        {/* Hero */}
        <section className="hero">
          <p className="hero-eyebrow">Evidence-first retrieval</p>
          <h1 className="hero-headline">
            It won't answer <em>until the evidence holds up.</em>
          </h1>
          <p className="hero-sub">
            Upload documents to build the case file, then put a question to the record. A LangGraph-orchestrated
            Evidence Agent checks what's retrieved before a Report Agent is ever allowed to answer — and retries
            retrieval on its own when the evidence comes up short.
          </p>
        </section>

        {/* Workspace */}
        <div className="workspace">
          {/* Case file panel */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">The Case File</h2>
              <span className="panel-count">{caseFiles.length} indexed</span>
            </div>
            <p className="panel-sub">PDF, DOCX, or TXT. Every file you add becomes searchable evidence.</p>

            <div
              className={`dropzone${isDragging ? " is-dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <p className="dropzone-label">Drop a document here, or</p>
              <span className="dropzone-file">choose a file to add to the record</span>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={(e) => submitFile(e.target.files?.[0])}
              />
            </div>

            {uploadStatus && (
              <p className={`upload-status${uploadStatus.error ? " is-error" : ""}`}>{uploadStatus.text}</p>
            )}

            {caseFiles.length > 0 ? (
              <ul className="case-list">
                {caseFiles.map((f) => (
                  <li className="case-item" key={f.id}>
                    <span className="case-item-name">{f.name}</span>
                    <span className="case-item-meta">{f.chunks} chunks</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-note">The case file is empty. Add a document to begin.</p>
            )}
          </section>

          {/* Interrogation panel */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Put a Question to the Record</h2>
            </div>
            <p className="panel-sub">Answered strictly from documents in the case file above.</p>

            <div className="question-row">
              <input
                className="question-input"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                placeholder="What does the record say about…"
              />
              <button className="btn btn-primary" style={{ marginTop: 0 }} onClick={handleAsk} disabled={loading}>
                {loading ? "Working" : "Ask"}
              </button>
            </div>

            {/* Trace rail — the live reasoning trace */}
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
                      <span className="trace-retry-badge">retried ×{liveRetries}</span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Findings */}
            {result ? (
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

                {result.sources?.length > 0 && (
                  <>
                    <p className="report-meta">Supporting evidence</p>
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
                  </>
                )}
              </div>
            ) : (
              !loading && <p className="report-empty">No question on the record yet.</p>
            )}
          </section>
        </div>

        <div className="footer">Agentic RAG System · LangGraph orchestration, evidence-verified answers</div>
      </div>
    </div>
  );
}