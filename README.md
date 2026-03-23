# DaVinci

## What DaVinci Is

DaVinci is a **persistent memory augment** for a locally-running LLM (via LM Studio).

It solves a fundamental problem: AI models have no memory between sessions. Every conversation starts from zero. DaVinci fixes this by storing, organising, and recalling knowledge on behalf of the LLM — making it behave like a persistent, learning assistant across sessions and reboots.

**DaVinci is not a standalone AI. It is the memory layer that gives an LLM continuity.**

---

## The Problem It Solves

- LLMs are stateless — they forget everything when a session ends
- You cannot natively train a specific intent or knowledge base into a local LLM
- Re-feeding context manually every session is slow, inconsistent, and lossy
- No existing tool provides persistent, structured, heuristic memory for local LLM use

---

## Long-Term Goal

Build a persistent AI assistant capable of producing **true-to-scale 3D models** (aircraft, race circuits) from:
- Technical specification documents
- Engineering blueprints
- Photographs

The assistant must accumulate and retain deep domain knowledge in 3D modeling, aeronautical engineering, and scale accuracy — knowledge that survives across every session.

**This is a phased goal. The memory system is Phase 1.**

---

## Architecture

### Memory Layer (`davinci/memory.py`)
- SQLite-backed persistent store — survives reboots
- `MemoryNode` — the core data unit with fields:
  - `content` — the raw memory text
  - `zoom_levels` — fractal detail levels (1=summary, 2=detail, 3=full spec)
  - `classification` — `core`, `boundary`, `decay`, `forget`
  - `frequency` — how often this memory has been accessed/reinforced
  - `created_at`, `last_accessed` — temporal tracking
  - `speaker`, `source`, `tags` — provenance tracking
  - `context_c` — fractal context parameter

### Fractal Decay (`davinci/fractals.py`)
Memory retention is modelled on Julia set escape-time dynamics:
- Frequently accessed memories decay slowly (high retention τ)
- Rarely accessed memories decay faster
- Memories reclassify: `core → boundary → decay → forget`
- This mimics human long-term vs short-term memory behaviour

**Why fractal decay?** For structured technical data (specs, measurements, part hierarchies), memories have natural importance gradients. A wingspan measurement accessed repeatedly during a modelling session should be retained far longer than a passing reference. The fractal model gives this behaviour mathematically without hard-coded rules.

### LLM Interface (`davinci/llm/`)
- Connects to any active model in LM Studio via its local API
- No model lock-in — works with whatever model is loaded
- Commands: `ingest`, `ask`, `chat`

### Public API (`davinci/interface/api.py`)
The `DaVinci` class exposes:
- `remember(content)` — store a memory
- `recall(id)` — retrieve by UUID
- `search(query)` — text search across memories
- `forget(classification)` — prune by class
- `decay()` — run the decay cycle
- `consolidate()` — merge/strengthen memories
- `merge_similar()` — deduplicate near-identical memories
- `stats()` — memory health overview

---

## Development Phases

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Persistent memory system | ✅ In progress |
| 2 | Deep ingestion of 3D modeling / domain knowledge | ⏳ Next |
| 3 | Interactive chat with persistent context | ⏳ Planned |
| 4 | Multimodal input (blueprints, images, PDFs) | ⏳ Future |
| 5 | 3D geometry generation (Blender scripting / gen models) | ⏳ Future |
| 6 | Scale validation against source specs | ⏳ Future |

---

## Design Principles

1. **Memory first** — without persistence, nothing else works
2. **LLM-agnostic** — DaVinci augments the LLM, it does not depend on a specific one
3. **Accumulative** — knowledge builds over time, sessions, and ingestion cycles
4. **Resilient** — memory survives reboots, session loss, and model swaps
5. **Solo-operable** — designed to be run and extended by a single non-technical operator using AI coding assistants

---

## Known Limitations (Current Stage)

- Search is text/keyword-based — semantic search not yet implemented
- `eval()` used in `MemoryNode.from_row()` for zoom_levels — known security risk, to be replaced with `json.loads()`
- No interactive chat mode — each `chat` call is stateless at CLI level
- No multimodal input yet (images, PDFs, blueprints)
- LM Studio must be running locally for LLM commands

---

## Context for AI Assistants

If you are an AI reading this to help develop DaVinci:

- **Do not simplify the memory schema** — it is designed for structured technical domain data, not general chatbot use
- **Do not remove fractal decay** — it is an intentional design choice for importance-weighted retention
- **The zoom level system is intentional** — it maps to detail levels in technical specifications (overview / detail / raw measurement)
- **The long-term goal is 3D model generation** — every design decision should be evaluated against whether it supports or hinders that goal
- **The developer cannot read or write code** — clarity, safety, and correctness matter more than cleverness
- **This README is the source of truth for design intent** — when in doubt, refer here

---

## Stack

- Python 3.10+
- SQLite (via `sqlite3`)
- LM Studio (local LLM host)
- Windows (`.bat` launcher included)

---

*DaVinci v0.5.2 — Designed by Claude Sonnet. Developed solo.*