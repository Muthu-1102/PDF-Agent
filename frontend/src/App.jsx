import { useState } from "react";

/**
 * Deliberately minimal UI. The point of this project is the Agentic RAG /
 * LangGraph backend -- this just gives you a way to upload files and ask
 * questions without curling the API by hand.
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleUpload() {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus("Uploading...");
    const res = await fetch("/upload/", { method: "POST", body: formData });
    const data = await res.json();
    setUploadStatus(
      res.ok
        ? `Indexed "${data.document_name}" (${data.chunks_indexed} chunks)`
        : `Error: ${data.detail}`
    );
  }

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setResult(null);
    const res = await fetch("/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setResult(data);
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Agentic RAG</h1>

      <section style={{ marginBottom: 32 }}>
        <h3>1. Upload a document</h3>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} style={{ marginLeft: 8 }}>
          Upload
        </button>
        <p>{uploadStatus}</p>
      </section>

      <section>
        <h3>2. Ask a question</h3>
        <input
          style={{ width: "70%" }}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask something about your uploaded docs..."
        />
        <button onClick={handleAsk} style={{ marginLeft: 8 }}>
          {loading ? "Thinking..." : "Ask"}
        </button>

        {result && (
          <div style={{ marginTop: 16 }}>
            <p>
              <strong>Answer:</strong> {result.answer}
            </p>
            <p>
              <strong>Confidence:</strong> {result.confidence} &nbsp;|&nbsp;
              <strong>Retries:</strong> {result.retries}
            </p>
            <p>
              <strong>Sources:</strong>
            </p>
            <ul>
              {result.sources?.map((s, i) => (
                <li key={i}>
                  {s.document} — page {s.page_number}, chunk {s.chunk_id} (score {s.score})
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}
