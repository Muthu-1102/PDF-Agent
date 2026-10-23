"""
Procedural synthetic dataset generator for PDF-Agent RAG evaluation.
Generates fictional documents, ground-truth facts, and a comprehensive question suite.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import docx

# Ensure backend is in sys.path to run the real chunker and loader
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(DOCS_DIR, exist_ok=True)


def create_pdf(filename: str, pages_text: List[str]) -> str:
    path = os.path.join(DOCS_DIR, filename)
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        # insert text with wrap
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()
    return path


def create_docx(filename: str, paragraphs: List[str], table_data: List[List[str]] = None) -> str:
    path = os.path.join(DOCS_DIR, filename)
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_data:
        table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
        for r_idx, row in enumerate(table_data):
            for c_idx, val in enumerate(row):
                table.cell(r_idx, c_idx).text = val
    doc.save(path)
    return path


def create_txt(filename: str, text: str, encoding: str = "utf-8") -> str:
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "w", encoding=encoding, errors="replace" if encoding == "utf-8" else "strict") as f:
        f.write(text)
    return path


def create_binary_file(filename: str, data: bytes) -> str:
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def generate_documents() -> Dict[str, str]:
    created = {}

    # 1. Travel Policy v1 (Old policy - superseded)
    p1 = (
        "VRCC Consortium Travel Policy Document Version 1.0 (Effective 2024).\n\n"
        "Section 1: Per Diem and Meals.\n"
        "Employees traveling on official VRCC business are entitled to a maximum daily meal per diem of $65 per day. "
        "Any meal expenditure exceeding $65 per day requires pre-authorization by the regional comptroller.\n\n"
        "Section 2: Lodging Allowances.\n"
        "The standard hotel nightly rate ceiling is fixed at $180 per night for domestic travel and $220 for international travel. "
        "Room service and minibar charges are strictly non-reimbursable under Policy 1.0.\n\n"
        "Section 3: Discretionary Spending.\n"
        "The discretionary project spending cap without executive review is $2,500 per trip. "
        "All receipts must be submitted to expense-audit@vaelen.internal within 14 calendar days."
    )
    created["vrcc_travel_policy_v1.pdf"] = create_pdf("vrcc_travel_policy_v1.pdf", [p1])

    # 2. Travel Policy v2 (Current policy - conflicting numbers)
    p2_page1 = (
        "VRCC Consortium Travel Policy Document Version 2.4 (Effective January 2026).\n"
        "Notice: This policy explicitly supersedes Version 1.0 in its entirety.\n\n"
        "Section 1: Per Diem and Subsistence.\n"
        "Under Policy Version 2.4, the updated daily meal per diem allowance is raised to exactly $92 per day. "
        "Breakfast is allocated $22, lunch $28, and dinner $42. Alcohol remains non-reimbursable.\n\n"
        "Section 2: Lodging Ceilings.\n"
        "The updated standard hotel ceiling is increased to $245 per night for all standard domestic tier-1 cities. "
        "For Tier-2 cities, the cap is $195 per night."
    )
    p2_page2 = (
        "Section 3: Executive Approval Thresholds.\n"
        "Discretionary travel spending authority is expanded: project directors can approve expenses up to $4,000 per trip. "
        "Expenses exceeding $4,000 require written clearance from Chief Financial Officer Marcus Vance.\n\n"
        "Section 4: Mileage and Electric Fleet.\n"
        "Employees driving company electric vehicles receive a charging stipend of $0.34 per mile. "
        "Car rental upgrades to premium tier require Medical Disability Form MED-7."
    )
    created["vrcc_travel_policy_v2.pdf"] = create_pdf("vrcc_travel_policy_v2.pdf", [p2_page1, p2_page2])

    # 3. Aethelgard Titan Drone Specs (Multi-page PDF)
    drone_p1 = (
        "Aethelgard Dynamics - Titan-X Heavy Autonomous Unmanned Aerial Vehicle Specifications.\n\n"
        "1. Propulsion and Battery Core:\n"
        "The Titan-X is powered by an advanced Lithium-Sulfur solid-core battery pack with an energy density of 420 Wh/kg. "
        "The battery system operates at a nominal bus voltage of 74.0 Volts and delivers 85 Amp-hours. "
        "Rotor architecture utilizes an octocopter coaxial-tilt layout with carbon-fiber reinforced blades."
    )
    drone_p2 = (
        "2. Payload Capacity and Aerodynamics:\n"
        "The maximum certified payload capacity of the Titan-X is exactly 18.4 kilograms. "
        "Total maximum takeoff weight (MTOW) is 46.8 kilograms. "
        "Under standard sea-level atmospheric conditions (ISA), the maximum continuous flight endurance is 142 minutes with a 5 kg sensor payload."
    )
    drone_p3 = (
        "3. Avionics and Communication Protocol:\n"
        "Primary telemetry is maintained via the proprietary Kestrel-Link 900 MHz redundant datalink with AES-256-GCM encryption. "
        "Operational ceiling is restricted to 4,500 meters above ground level. "
        "In the event of lost telemetry exceeding 45 seconds, the emergency return-to-base sequence RTB-ALPHA is automatically engaged."
    )
    created["aethelgard_titan_drone_specs.pdf"] = create_pdf("aethelgard_titan_drone_specs.pdf", [drone_p1, drone_p2, drone_p3])

    # 4. Long 52-page PDF: Zephyros Avionics Handbook
    long_pages = []
    for page_idx in range(1, 53):
        code_num = 1000 + page_idx * 17
        freq_val = 120.0 + (page_idx * 0.45)
        text = (
            f"Zephyros Aerospace Avionics Systems Engineering Manual — Chapter {page_idx}.\n\n"
            f"Subsection {page_idx}.1: Technical Diagnostics for Module ZX-{code_num}.\n"
            f"The primary telemetry carrier frequency for subsystem {page_idx} is calibrated to {freq_val:.2f} MHz. "
            f"During nominal sub-orbital transit, coolant flow rate must be maintained at {page_idx * 1.5 + 10:.1f} liters per minute. "
            f"The safety interlock threshold for voltage regulator VREG-{page_idx} is set at {24.0 + (page_idx * 0.1):.1f} V.\n\n"
            f"Subsection {page_idx}.2: Standard Operating Procedure.\n"
            f"Inspect pressure seal GASKET-{page_idx} every 250 operating cycles. "
            f"If differential pressure drops below 1.25 bar, issue alarm CODE-{code_num}-WARN and isolate the purge valve."
        )
        long_pages.append(text)
    created["zephyros_avionics_handbook.pdf"] = create_pdf("zephyros_avionics_handbook.pdf", long_pages)

    # 5. Security Protocol DOCX with a Table (Testing H10)
    docx_paras = [
        "Nyx Protocol Cyber Security Incident Classification and Response Playbook.",
        "This document establishes the mandatory response procedures for unauthorized access, cryptographic leakage, and infrastructure anomalies.",
        "General Rule: All incidents must be logged in SIEM within 5 minutes of automated alert generation.",
        "The following classification matrix defines incident severity and escalation responsibilities:"
    ]
    table = [
        ["Severity", "Incident Type", "SLA Response", "Escalation Lead", "Pager Code"],
        ["Critical-1", "Rogue Key Generation", "15 minutes", "Dr. Aris Thorne", "PK-9912"],
        ["Major-2", "DDoS Exceeding 50Gbps", "30 minutes", "Elena Rostova", "PK-7740"],
        ["Moderate-3", "Failed Auth Burst (>500)", "60 minutes", "David Kim", "PK-4419"],
        ["Minor-4", "Certificate Expiry Warning", "24 hours", "Sarah Jenkins", "PK-1102"]
    ]
    docx_paras.append("Post-incident forensic images must be preserved in immutable S3 storage for a minimum of 7 years under regulatory mandate SEC-804.")
    created["nyx_security_incident_response.docx"] = create_docx("nyx_security_incident_response.docx", docx_paras, table)

    # 6. Project Chronos Research (TXT) - Position-sensitive chunks for H1 and H2
    # Chunk A has target fact in first 30 words. Chunk B has target fact at words 465-495 (tokens > 600).
    filler_200 = (
        "The thermal regulation subsystem undergoes rigorous cyclic environmental vibration testing. "
        "Standard laboratory conditions require ambient humidity below forty percent and nitrogen atmosphere. "
        "Telemetry data streams are digitized via sixteen bit analog converters with oversampling factors of eight. "
        "System stability requires continuous impedance matching across all high frequency transmission junctions. "
        "Calculated mean time between failures exceeds fifty thousand operating hours under nominal load profiles. "
        "Signal attenuation across the wave guide assembly is monitored via directional couplers calibrated annually. "
        "Electromagnetic shielding conforms to Class B military aerospace specifications with double braided copper mesh. "
        "Grounding straps must maintain resistance below two milliohms across all structural chassis joints. "
        "Cooling radiators utilize closed loop dielectric liquid fluorocarbon coolant circulating at variable velocities. "
        "Pressure relief diaphragms burst at precisely three point five bar to prevent internal casing rupture. "
    ) * 2  # ~220 words

    filler_250 = (
        "Operational telemetry is recorded across redundant solid state non volatile memory banks. "
        "All data packets are timestamped with microsecond precision derived from an onboard rubidium atomic clock standard. "
        "Field programmable gate arrays manage sensor fusion algorithms processing multi axis inertial measurement unit data. "
        "Diagnostic routines execute every fifty milliseconds checking memory parity and bus contention. "
        "Software watchdogs reset the primary flight controller if handshake pulses fail for two consecutive cycles. "
        "Thermal insulation consists of multilayer aluminized mylar blankets separated by dacron netting. "
        "Optical sensor windows feature sapphire glass with broadband anti reflective diamond like carbon coating. "
        "Power distribution units employ silicon carbide metal oxide semiconductor field effect transistors. "
        "Harmonic distortion in the alternating current bus is actively suppressed by active digital filtering networks. "
        "Telemetry downlink packets are framed using high level data link control protocols with cyclic redundancy checks. "
    ) * 2  # ~240 words

    # Fact at START:
    start_fact_chunk = (
        "START_POSITION_TEST_RECORD: The Chronos sub-harmonic frequency is calibrated to exactly 418.75 MHz by senior physicist Dr. Julian Vance. "
        + filler_200 + " " + filler_250
    )
    # Ensure it's approximately 490 words so it stays as 1 chunk in the 500-word chunker
    start_words = start_fact_chunk.split()[:490]
    chunk_start_text = " ".join(start_words)

    # Fact at END:
    end_fact_chunk = (
        filler_200 + " " + filler_250 + " "
        + "END_POSITION_TEST_RECORD: The thermal dissipation buffer threshold is rated at exactly 1340 Joules per Kelvin by lead thermodynamicist Dr. Cynthia Cruz."
    )
    end_words = end_fact_chunk.split()[:490]
    chunk_end_text = " ".join(end_words)

    chronos_full_text = (
        "PROJECT CHRONOS ADVANCED PROPULSION AND QUANTUM DYNAMICS TECHNICAL BRIEFING.\n\n"
        "SECTION 1: SUB-HARMONIC CALIBRATION.\n"
        + chunk_start_text + "\n\n"
        "SECTION 2: THERMAL DISSIPATION BUFFER.\n"
        + chunk_end_text + "\n\n"
        "SECTION 3: SUMMARY RECOMMENDATIONS.\n"
        "All tests conducted at the White Sands Proving Ground confirmed resonance stability within plus or minus 0.05 percent. "
        "Next quarterly review is scheduled for December 2026."
    )
    created["project_chronos_research.txt"] = create_txt("project_chronos_research.txt", chronos_full_text)

    # 7. Aurora Quantum Compiler (PDF)
    aurora_p1 = (
        "Aurora Quantum Compiler Architecture Whitepaper (Release 4.1).\n\n"
        "The Aurora Quantum Compiler targets the proprietary Q-Alpha 7 superconducting qubit lattice. "
        "The nominal decoherence window of the processing nodes is precisely 840 microseconds under cryogenic shielding. "
        "The primary cryostat dilution refrigerator maintains a continuous base operating temperature of 12.4 millikelvin."
    )
    aurora_p2 = (
        "Compilation and Optimization Passes:\n"
        "Aurora implements the Helios-9 gate synthesis algorithm, reducing two-qubit CNOT gate counts by 38.6 percent. "
        "Maximum supported physical qubit register size in the current deployment is 128 topological qubits. "
        "Compiler errors and syntax exceptions are logged to sys-quantum@vaelen.internal."
    )
    created["aurora_quantum_compiler.pdf"] = create_pdf("aurora_quantum_compiler.pdf", [aurora_p1, aurora_p2])

    # 8. Orion Fleet Logistics (DOCX - Paragraphs)
    orion_paras = [
        "Orion Autonomous Fleet Logistics and Regional Distribution Report.",
        "Fleet distribution across regional distribution centers is organized as follows:",
        "Northport Logistics Hub: Operates a fleet of 45 autonomous electric transport vans and 12 heavy haulers.",
        "Southhaven Distribution Depot: Operates 32 autonomous electric transport vans and 8 heavy haulers.",
        "Eastridge Terminal: Operates 58 autonomous electric transport vans and 19 heavy haulers.",
        "Westbay Maritime Complex: Operates 74 autonomous electric transport vans and 26 heavy haulers.",
        "Total network capacity accommodates 4,200 metric tons of freight daily with average delivery turnaround of 4.2 hours."
    ]
    created["orion_fleet_logistics.docx"] = create_docx("orion_fleet_logistics.docx", orion_paras)

    # 9. Scanned Archive Receipt (Image-only PDF / zero text extractable)
    # Create a PDF page with an empty drawn rect, no text
    scanned_path = os.path.join(DOCS_DIR, "scanned_archive_receipt.pdf")
    doc_scan = fitz.open()
    p_scan = doc_scan.new_page()
    p_scan.draw_rect(fitz.Rect(100, 100, 400, 400), color=(0.5, 0.5, 0.5), fill=(0.9, 0.9, 0.9))
    doc_scan.save(scanned_path)
    doc_scan.close()
    created["scanned_archive_receipt.pdf"] = scanned_path

    # 10. Latin-1 encoded TXT file (Testing H10 non-UTF-8)
    latin1_path = os.path.join(DOCS_DIR, "latin1_legacy_notes.txt")
    with open(latin1_path, "wb") as f:
        f.write("Legacy Maintenance Log for Societe Generale Turbines.\nTemperature: 85 deg C. Cout: 1500 GBP.\n".encode("latin-1") + bytes([0xE9, 0xE0, 0xF4]))
    created["latin1_legacy_notes.txt"] = latin1_path

    # 11. Empty TXT file
    empty_path = os.path.join(DOCS_DIR, "empty_document.txt")
    with open(empty_path, "w", encoding="utf-8") as f:
        pass
    created["empty_document.txt"] = empty_path

    # 12. Dotless filename
    dotless_path = os.path.join(DOCS_DIR, "unsupported_dotless_doc")
    with open(dotless_path, "w", encoding="utf-8") as f:
        f.write("This file has no extension.")
    created["unsupported_dotless_doc"] = dotless_path

    return created


def run_chunking_and_map_gold(questions: List[Dict[str, Any]], docs_created: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Runs the app's real loader and chunker on the created documents,
    indexes all chunks into a dictionary, and maps gold evidence strings to actual (doc_name, chunk_index) IDs.
    """
    indexed_chunks = {}  # doc_name -> List[Tuple[chunk_idx, page_number, chunk_text]]
    
    # Files to index in standard knowledge base (excluding edge-case failure files)
    standard_files = [
        "vrcc_travel_policy_v1.pdf",
        "vrcc_travel_policy_v2.pdf",
        "aethelgard_titan_drone_specs.pdf",
        "zephyros_avionics_handbook.pdf",
        "nyx_security_incident_response.docx",
        "project_chronos_research.txt",
        "aurora_quantum_compiler.pdf",
        "orion_fleet_logistics.docx"
    ]

    for fname in standard_files:
        fpath = docs_created[fname]
        ftype = fname.split(".")[-1]
        pages = load_document(fpath, ftype)
        chunks = chunk_document(pages)
        indexed_chunks[fname] = []
        for idx, (page_num, chunk_text) in enumerate(chunks):
            indexed_chunks[fname].append({
                "chunk_index": idx,
                "page_number": page_num,
                "text": chunk_text
            })

    # Save indexed chunks metadata for reference
    with open(os.path.join(os.path.dirname(__file__), "indexed_chunks_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(indexed_chunks, f, indent=2)

    # For each question with a gold_evidence_string, find which chunk contains it
    mapped_questions = []
    for q in questions:
        q_copy = dict(q)
        gold_chunks = []
        evidence_str = q.get("gold_evidence_string")
        target_doc = q.get("gold_document")

        if evidence_str and target_doc:
            if isinstance(target_doc, str):
                target_docs = [target_doc]
            else:
                target_docs = target_doc

            evidence_strings = [evidence_str] if isinstance(evidence_str, str) else evidence_str

            for doc_name in target_docs:
                chunks = indexed_chunks.get(doc_name, [])
                for ev in evidence_strings:
                    for c in chunks:
                        if ev.lower() in c["text"].lower():
                            chunk_ref = f"{doc_name}_chunk_{c['chunk_index']}"
                            if chunk_ref not in gold_chunks:
                                gold_chunks.append({
                                    "document_name": doc_name,
                                    "chunk_index": c["chunk_index"],
                                    "page_number": c["page_number"],
                                    "ref_id": chunk_ref
                                })

        q_copy["gold_chunks"] = gold_chunks
        mapped_questions.append(q_copy)

    return mapped_questions, indexed_chunks


if __name__ == "__main__":
    docs = generate_documents()
    print(f"Generated {len(docs)} documents in {DOCS_DIR}")
