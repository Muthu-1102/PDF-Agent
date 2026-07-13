Architecture.md

Agentic RAG System Architecture

Version: 1.0
Purpose: Demonstrate an advanced Agentic Retrieval-Augmented Generation
(Agentic RAG) system using LangGraph, emphasizing autonomous retrieval
validation, iterative reasoning, and evidence-based response generation.

------------------------------------------------------------------------

1. Project Overview

This project is an Agentic RAG System capable of ingesting multiple
documents during an active conversation and answering user questions
strictly from the uploaded knowledge base.

Unlike traditional RAG systems that retrieve Top-K documents once and
immediately generate an answer, this system introduces autonomous agents
that evaluate retrieval quality, retry retrieval when necessary, and
generate responses only when sufficient evidence exists.

The primary objective of this project is to demonstrate:

-   LangGraph orchestration
-   Multi-agent collaboration
-   Evidence-driven retrieval
-   Retrieval verification
-   Stateful workflow execution
-   Dynamic knowledge base updates

------------------------------------------------------------------------

2. Technology Stack

  Layer                Technology
  -------------------- ----------------------------------------
  Frontend             React
  Backend API          FastAPI
  Agent Framework      LangGraph
  LLM                  Groq API
  Embedding Model      sentence-transformers/all-MiniLM-L6-v2
  Vector Database      ChromaDB
  File Processing      PyMuPDF, python-docx
  Conversation State   LangGraph State
  Language             Python

------------------------------------------------------------------------

3. High-Level Architecture

                        React UI
                            |
            Upload Files / Ask Questions
                            |
                         FastAPI
                            |
                  LangGraph Orchestrator
                    (RAG Controller)
                     /             \
          Upload Workflow    Query Workflow

------------------------------------------------------------------------

4. System Architecture

USER | +– Upload Documents –> Upload Pipeline –> Document Processing –>
Chunk Generation –> Embedding Model –> ChromaDB | +– Ask Question —–>
Query Pipeline –> LangGraph Workflow –> Retrieve Candidate Chunks |
Evidence Verification | Enough Evidence? /
YES NO | | Report Generation Improve Retrieval   / Final Response

------------------------------------------------------------------------

5. Overall Workflow

The system consists of two independent workflows:

1.  Document Upload Workflow
2.  Query Execution Workflow

The upload workflow continuously expands the knowledge base.

The query workflow is responsible for intelligent retrieval and response
generation.

------------------------------------------------------------------------

6. Upload Workflow

User Upload -> File Validation -> Document Loader -> Text Extraction ->
Text Cleaning -> Chunking -> Embedding Generation -> Store Embeddings ->
Update Vector Database

Supported file types: - PDF - DOCX - TXT

Metadata stored: - Document ID - Document Name - Page Number - Chunk
ID - Upload Timestamp - Chunk Index

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Stored in ChromaDB: - Embedding - Chunk Text - Metadata

------------------------------------------------------------------------

7. Query Workflow

User Question -> RAG Orchestrator -> Retriever -> Evidence Agent ->
Report Agent -> Final Answer

------------------------------------------------------------------------

8. LangGraph Architecture

START | Receive Question | Retrieve Candidate Chunks | Evidence
Evaluation | Enough Evidence? | | YES NO | | Generate Retry Retrieval
Report | | | +—— END –+

------------------------------------------------------------------------

9. Agent Responsibilities

RAG Orchestrator - Controls workflow - Maintains state - Coordinates
agents - Handles retries - Never generates answers

Evidence Agent - Evaluates retrieval quality - Checks whether evidence
is sufficient - Requests new retrieval when needed

Inputs: - User Question - Retrieved Chunks - Similarity Scores -
Metadata

Outputs: - Evidence Status - Trusted Chunks - Retry Required

Report Agent - Generates answer only from verified evidence - Produces
citations - Returns confidence score

Outputs: - Answer - Supporting Documents - Page Numbers - Confidence

------------------------------------------------------------------------

10. Retrieval Retry Loop

Initial Retrieval -> Evidence Evaluation -> Enough? /
YES NO | | Answer Improve Retrieval | Retrieve Again | Evidence
Evaluation

Possible improvements: - Increase Top-K - Expand semantic search - Relax
similarity threshold - Merge additional chunks - Query rewriting
(future)

------------------------------------------------------------------------

11. LangGraph State

-   Question
-   Conversation History
-   Retrieved Chunks
-   Retrieved Metadata
-   Similarity Scores
-   Trusted Chunks
-   Retry Count
-   Current Top-K
-   Evidence Status
-   Generated Answer
-   Sources

------------------------------------------------------------------------

12. Knowledge Base

Stores: - Embedding Vector - Chunk Text - Document ID - Document Name -
Page Number - Chunk ID - Upload Timestamp

Supports incremental document uploads.

------------------------------------------------------------------------

13. Conversation Flow

Upload A -> Indexed

Upload B -> Indexed

Ask Question -> Search A + B

Upload C -> Indexed

Next Question -> Search A + B + C

------------------------------------------------------------------------

14. Design Principles

-   Evidence First
-   Retrieval Before Generation
-   Stateless FastAPI
-   Stateful LangGraph
-   Modular Agents
-   Extensible Architecture

------------------------------------------------------------------------

15. Why LangGraph?

Traditional RAG:

Retrieve -> Generate -> Return

Agentic RAG:

Retrieve -> Evaluate Evidence -> Retry if Needed -> Generate -> Return

Supports loops, state management, conditional execution, and agent
coordination.

------------------------------------------------------------------------

16. Future Enhancements

-   Hybrid Retrieval
-   Query Rewriting
-   Citation Verification
-   Hallucination Detection
-   Parent-Child Retrieval
-   Cross-Encoder Reranking
-   Document Summarization
-   Multi-modal Support
-   Streaming Responses
-   Multi-user Workspaces

------------------------------------------------------------------------

17. Summary

This architecture demonstrates a production-oriented Agentic RAG system
using React, FastAPI, LangGraph, Groq, ChromaDB, and
sentence-transformers/all-MiniLM-L6-v2. It separates ingestion,
retrieval, evidence validation, and report generation into modular
components while showcasing stateful orchestration, iterative retrieval,
and evidence-grounded answer generation.
