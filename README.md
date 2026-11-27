# 🛡️ PhishGuard v3.0

**Context-Gated Multimodal Web Phishing Detection Without Headless Browsers or Native Deep-Learning Dependencies**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%2B%20Calibrated%20LogReg-success)](https://lightgbm.readthedocs.io/)
[![ROC-AUC](https://img.shields.io/badge/Holdout%20ROC--AUC-0.9904-brightgreen)](#6-benchmark-results--ablation-study)
[![Unseen Audit](https://img.shields.io/badge/1k%20Unseen%20ROC--AUC-0.9877-brightgreen)](#62-external-1000-domain-unseen-shard-audit-files15)
[![Zero DLL](https://img.shields.io/badge/Runtime-Zero%20Chromium%20%2F%20Zero%20DLL-blue)](#1-overview--motivation)

---

## 1. Overview & Motivation

Modern phishing campaigns evade single-modality lexical filters using typosquatting, homoglyph domains, subdomain nesting, cross-origin credential forms, and image-cloaked login pages. Conversely, vision-based detectors rely on headless Chromium browsers (Playwright/Selenium) and heavy C++/PyTorch deep-learning libraries that trigger Windows Smart App Control blocks and introduce multi-second inference latency.

**PhishGuard v3.0** solves both problems with a lightweight, pure-Python **five-signal context-gated ensemble** packaged as an asynchronous **FastAPI** microservice (`< 10 MB` serialized artifact) that delivers sub-second CPU predictions (`< 1 s`) with **zero headless browser or native C++ DLL dependencies**.

### Core Capabilities
* **Five-Signal Multimodal Fusion:** Combines **URL Lexical**, **URL Subword N-Gram (`3–5`)**, **DOM Behavioral**, **Semantic HTML**, and a pure-Python **Visual Brand-Asset Integrity Analyzer**.
* **Neutral-Prior (`0.50`) Modality Gating:** Prevents calibration collapse when target websites return `HTTP 403` Web Application Firewall (WAF) challenges or connection timeouts.
* **Domain-Disjoint Generalization:** Trained and validated using strict `GroupShuffleSplit` by registered apex domain (`tldextract`) plus synthetic adversarial brand-spoofing augmentation.
* **Explainable AI (XAI) Telemetry:** Returns per-modality risk probabilities, human-readable risk drivers, HTTP fetch status, and live `/favicon.ico` SHA-256 digests.

---

## 2. Project Scope & Boundaries

| Dimension | In-Scope (Supported) | Out-of-Scope (Intentional Boundaries) |
| :--- | :--- | :--- |
| **Threat Vector** | HTTP/HTTPS web URLs and static HTML/DOM payloads targeting credential theft and brand impersonation. | Email transport/headers (`SPF`/`DKIM`/`DMARC`), raw email attachments, SMS (smishing), and voice phishing. |
| **Visual Inspection** | Pure-Python asset-level integrity: cross-domain logo hotlinking, `/favicon.ico` SHA-256 byte fingerprinting, and cloaked image-form detection. | Full-page screenshot rendering via headless Chromium/Playwright or pixel-level CNN/ViT inference. |
| **Client-Side Code** | Static HTML/DOM structure and inline script/iframe counts. | Dynamic JavaScript execution (`eval()`, SPA hydration) or binary malware sandboxing. |
| **Runtime Envelope** | Sub-second CPU inference on Windows and Linux without native deep-learning DLLs. | Paid real-time WHOIS/RDAP or Certificate Transparency API feeds (planned for `v3.2`). |

---

## 3. System Architecture

PhishGuard v3.0 uses a two-stage stacked architecture where four calibrated base experts and a pure-Python visual integrity analyzer feed risk probabilities and observability flags (`html_available`) into a LightGBM context-gated meta-learner:

```text
Target URL ──┬──► [1] URL Lexical Expert (Calibrated LightGBM: 28 structural/entropy features) ──┐
             ├──► [2] URL Subword Expert (Calibrated LogReg: char_wb 3–5 n-gram TF-IDF) ─────────┼──► Context-Gated
             │                                                                                    │    Meta-Learner ──► Calibrated Risk
             └──► Async HTTP Fetch ──┬── (HTTP 200) ──► [3] DOM Behavioral Expert (LightGBM) ────┤    (LightGBM)       + XAI Drivers
                                     │              ├── [4] Semantic HTML Expert (LogReg) ───────┤
                                     │              └── [5] Visual Brand-Asset Analyzer ─────────┤
                                     └── (403 / Timeout) ──► Neutral Prior Fallback (0.50) ──────┘
```

### Pure-Python Visual Brand-Asset Integrity (Module 5)
Instead of launching a browser to capture screenshots, Module 5 inspects visual assets directly from the HTTP layer in milliseconds:
1. **Cross-Domain Brand Asset Hotlinking:** Flags `<img>` and icon tags loading logos from protected corporate asset servers (`BRAND_ASSET_DOMAINS`) onto unauthorized apex domains, while filtering out generic web infrastructure CDNs (`gstatic.com`, `googleapis.com`).
2. **Cryptographic Favicon Fingerprinting:** Computes the live **SHA-256 digest** of `/favicon.ico` bytes to catch locally re-hosted brand icons.
3. **Cloaked Image-Form Detection:** Flags low-text pages (`< 90` visible characters) pairing `<img>` overlays with `<input>` credential fields.

---

## 4. Quickstart & Local Hosting

### 4.1 Installation

```powershell
git clone https://github.com/ShubhrGupta/phishguard.git
cd phishguard

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 4.2 Run Locally (`localhost:8000`)

```powershell
uvicorn app.main:app --reload
```
Open **`http://127.0.0.1:8000`** in your browser for the interactive scanner UI, or **`http://127.0.0.1:8000/docs`** for the Swagger OpenAPI documentation.

### 4.3 Expose on Local Wi-Fi / LAN or Public Tunnel
* **Local Wi-Fi / LAN:**
  ```powershell
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
* **Instant Public HTTPS URL (via Cloudflare Quick Tunnel):**
  ```powershell
  cloudflared tunnel --url http://localhost:8000
  ```

---

## 5. API Usage (`POST /api/analyze`)

### Request
```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/library/asyncio.html"}'
```

### Example Response
```json
{
  "url": "https://docs.python.org/3/library/asyncio.html",
  "apex_domain": "python.org",
  "phishing_probability": 0.0001,
  "verdict": "SAFE",
  "http_status": 200,
  "html_available": true,
  "modality_scores": {
    "url_lexical": 0.0214,
    "url_subword": 0.0185,
    "dom_behavioral": 0.0310,
    "semantic_html": 0.0122,
    "visual_integrity": 0.0000
  },
  "favicon_sha256": "a1b2c3d4e5f6...",
  "risk_drivers": [
    "Verified apex domain with clean lexical and DOM profile"
  ]
}
```

---

## 6. Benchmark Results & Ablation Study

### 6.1 25,000-Row Apex-Domain-Disjoint Holdout (`4,987` rows / `4,270` unseen apex domains)
Trained on `PhreshPhish` (`train` shards `0–14`) grouped strictly by registered apex domain (`GroupShuffleSplit`) so zero domains overlap between train and validation sets:

| Model Configuration | ROC-AUC | F1-Score | Precision | Recall | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| URL Lexical Expert Only | 0.9612 | 0.8945 | 0.9120 | 0.8776 | 0.0741 |
| URL Subword N-Gram Expert Only | 0.9688 | 0.9061 | 0.9245 | 0.8884 | 0.0685 |
| DOM Behavioral Expert Only | 0.9415 | 0.8630 | 0.8890 | 0.8385 | 0.0912 |
| Semantic HTML Text Expert Only | 0.9540 | 0.8812 | 0.9034 | 0.8601 | 0.0819 |
| **PhishGuard v3.0 (Fused + Context Gating)** | **0.9904** | **0.9492** | **0.9627** | **0.9360** | **0.0382** |

### 6.2 External 1,000-Domain Unseen Shard Audit (`files[15:]`)
Evaluated on `500` phishing and `500` benign domains completely disjoint from all development shards:

| Configuration | ROC-AUC | F1-Score | Precision | Recall | FPR | FNR | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Zero-Fill Missing HTML + Raw CDNs) | 0.9466 | 0.6530 | 0.9137 | 0.5080 | 4.80% | 49.20% | 0.2621 |
| **PhishGuard v3.0 (Neutral `0.50` Prior + CDN Fix)** | **0.9877** | **0.9319** | **0.9337** | **0.9300** | **6.60%** | **7.00%** | **0.0586** |

### 6.3 Live Production Web Audit (`14 / 15` Pass Rate — `93.33%`)
* **100% Pass Rate on `HTTP 403` WAF & Timeout Sites:** Sites returning `HTTP 403` (`stackoverflow.com`, `nytimes.com`) or `ReadTimeout` (`irctc.co.in`) cleanly fall back to the `0.50` neutral prior with `0.00%` false-positive risk.
* **Known Edge Case (`canva.com`):** Cloudflare bot challenges that return `HTTP 200` alongside a JavaScript verification interstitial (*"Verify you are human"*) are currently parsed as page HTML, causing a false positive (`99.96%`) on `https://www.canva.com/`.

---

## 7. Operational Guardrails & Roadmap

### Recommended Deployment Policy Tiers
* **Auto-Allow (`< 0.15`):** Low-risk URLs cleared immediately.
* **Analyst Review / Warning Banner (`0.15 – 0.85`):** Surface XAI risk drivers with a user override option to prevent lockouts on legitimate B2B portals (`6.60%` unseen FPR).
* **High-Confidence Block (`> 0.85`):** Strong multimodal phishing consensus.

### Roadmap
* **v3.1:** Signature detector for `HTTP 200` Cloudflare/Akamai bot-challenge interstitials (`cf-browser-verification`) to route them to the `0.50` neutral prior, plus SaaS root-domain vs. UGC path tiering (`canva.com` vs. `canva.com/design/...`).
* **v3.2:** Asynchronous RDAP/WHOIS domain-age and TLS Certificate Transparency metadata enrichment.
* **v4.0:** Pure-Python static JavaScript de-obfuscation (`atob`/`eval` inspection) and SQLite/PostgreSQL active-learning feedback loop.

---

## 8. Repository Structure

```text
phishguard/
├── app/
│   └── main.py                  # FastAPI microservice, async HTTP fetcher & context gating
├── models/
│   └── phishguard_bundle.joblib # Serialized 4-expert + meta-learner bundle (< 10 MB)
├── reports/
│   └── figures/                 # Ablation, ROC, calibration & SHAP diagnostic plots
├── requirements.txt             # Pure-Python dependency specification
└── README.md                    # Project documentation
```