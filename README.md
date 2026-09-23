# Prakriti-Niti AI (प्रकृति-नीति)

> **Indian Environmental Policy Scholar & Interactive RAG Assistant**

Prakriti-Niti AI is a conversational research tool built on top of scoping research evaluating conversational AI and Retrieval-Augmented Generation (RAG) frameworks for environmental policy governance in India. 

The application utilizes a grounded multi-tier evidence taxonomy to address regulatory nuances, domain-specific terminologies, and systemic policy workflows.

---

## Architecture & Evidence Taxonomy

The assistant operates across a structured three-tier policy evidence framework:

* **Tier 1: INDIA DIRECT (SukhaRakshak RAG)**  
  Direct environmental and disaster mitigation frameworks in India (e.g., drought management, local conservation guidelines).
* **Tier 2: INDIA ADJACENT (Bhashini / PARIVESH e-Gov)**  
  Broader Indian public AI infrastructure and administrative e-governance systems.
* **Tier 3: INTERNATIONAL TRANSFERABLE (Climate & Policy RAG)**  
  Comparative global environmental governance models and cross-border climate RAG architectures.

---

## Tech Stack

* **Backend:** FastAPI (Python 3.10+)
* **LLM Engine:** Google Gemini API (`gemini-1.5-flash` / `gemini-1.5-pro`)
* **Frontend:** Responsive Single-Page Application (HTML5, JavaScript)
* **Styling:** Tailwind CSS (via Tailwind Play CDN & `@tailwindcss/typography`) with custom earthy color schemes and dark-mode support
* **Markdown & Sanitation:** `marked.js` with `DOMPurify` for secure client-side rendering

---

## Project Structure

```text
Prakriti-Niti-AI/
├── backend/
│   ├── app.py              # FastAPI server, LLM integration, and route handlers
│   ├── index.html          # Interactive Scholar UI
│   ├── requirements.txt    # Python dependencies (fastapi, uvicorn, google-genai, etc.)
│   └── .env.example        # Environment variable template
├── data/
│   └── EMP_TT_ 1.pdf       # Core environmental policy research corpus (ignored by git)
├── .gitignore              # Excludes secrets, venv, and local data
└── README.md
