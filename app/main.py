import hashlib
import ipaddress
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import joblib
import httpx
import tldextract
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from features import (
    extract_url_features,
    extract_dom_features,
    extract_clean_text,
    normalize_url_for_subwords,
    extract_modality_integrity,
)

app = FastAPI(title="PhishGuard 2.1 Multimodal + Visual Asset Integrity API", version="2.1.0")

BUNDLE_PATH = ROOT_DIR / "models" / "phishguard_bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)

url_model = bundle["url_model"]
char_model = bundle["char_model"]
char_tfidf = bundle["char_tfidf"]
dom_model = bundle["dom_model"]
txt_model = bundle["txt_model"]
word_tfidf = bundle["tfidf"]
meta_model = bundle["meta_model"]

VERIFIED_APEX_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "microsoft.com", "live.com",
    "office.com", "apple.com", "icloud.com", "paypal.com", "amazon.com",
    "amazon.in", "netflix.com", "facebook.com", "instagram.com", "linkedin.com",
    "github.com", "chase.com", "wellsfargo.com", "bankofamerica.com",
    "binance.com", "coinbase.com", "adobe.com", "dropbox.com",
    "dhl.com", "fedex.com", "usps.com", "sbi.co.in", "hdfcbank.com", "icicibank.com"
}

# Official brand CDN & static asset domains often hotlinked by phishing kits
BRAND_ASSET_DOMAINS = {
    "paypal.com": "PayPal",
    "paypalobjects.com": "PayPal",
    "microsoft.com": "Microsoft",
    "msftauth.net": "Microsoft",
    "aadcdn.msauth.net": "Microsoft",
    "office.com": "Microsoft",
    "apple.com": "Apple",
    "mzstatic.com": "Apple",
    "google.com": "Google",
    "linkedin.com": "LinkedIn",
    "licdn.com": "LinkedIn",
    "facebook.com": "Facebook",
    "fbcdn.net": "Facebook",
    "amazon.com": "Amazon",
    "media-amazon.com": "Amazon",
    "ssl-images-amazon.com": "Amazon",
    "chase.com": "Chase",
    "wellsfargo.com": "Wells Fargo",
    "bankofamerica.com": "Bank of America",
    "dhl.com": "DHL",
    "fedex.com": "FedEx",
    "dropbox.com": "Dropbox",
    "dropboxstatic.com": "Dropbox",
    "adobe.com": "Adobe",
    "adobelogin.com": "Adobe",
}

USER_CONTENT_SUBDOMAINS = {"sites", "storage", "docs", "drive", "pages", "usercontent"}


class AnalyzeRequest(BaseModel):
    url: str
    html: str | None = None
    fetch_live: bool = True


def get_apex_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def check_verified_brand_domain(url: str) -> bool:
    clean = url.strip().lower()
    if not clean.startswith("https://"):
        return False
    ext = tldextract.extract(clean)
    apex = f"{ext.domain}.{ext.suffix}".lower()
    sub_parts = set(ext.subdomain.lower().split(".")) if ext.subdomain else set()
    if sub_parts & USER_CONTENT_SUBDOMAINS:
        return False
    # Allow all regional Google search domains (google.com, google.hr, google.co.in, etc.)
    if ext.domain.lower() == "google" and ext.suffix:
        return True
    return apex in VERIFIED_APEX_DOMAINS


def is_local_or_private_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


async def analyze_visual_brand_assets(
    page_url: str, html: str, fetch_live: bool, is_verified_brand: bool
) -> dict:
    """
    Option 3A: Analyzes favicon fingerprint, cross-domain brand asset hotlinking,
    and image-cloaked zero-text phishing layouts.
    """
    result = {
        "visual_risk_score": 0.0,
        "favicon_sha256": None,
        "hotlinked_brands": [],
        "image_cloaked_suspected": False,
        "inline_data_images": 0,
    }
    if is_verified_brand or not html:
        return result

    page_apex = get_apex_domain(page_url)
    soup = BeautifulSoup(html, "lxml")

    # 1. Inspect favicon, og:image, and img tags for cross-domain brand asset hotlinking
    asset_urls = []
    favicon_href = None
    for link in soup.find_all("link", href=True):
        rels = [r.lower() for r in (link.get("rel") or [])]
        if any("icon" in r for r in rels):
            href = link["href"].strip()
            asset_urls.append(href)
            if not favicon_href:
                favicon_href = href

    for meta in soup.find_all("meta", content=True):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        if prop in {"og:image", "twitter:image"}:
            asset_urls.append(meta["content"].strip())

    inline_data_imgs = 0
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if src.startswith("data:image"):
            inline_data_imgs += 1
        else:
            asset_urls.append(src)

    result["inline_data_images"] = inline_data_imgs

    hotlinked = set()
    for raw_asset in asset_urls:
        if raw_asset.startswith("data:"):
            continue
        abs_asset = urljoin(page_url, raw_asset)
        asset_apex = get_apex_domain(abs_asset)
        if asset_apex and asset_apex != page_apex and asset_apex in BRAND_ASSET_DOMAINS:
            hotlinked.add(BRAND_ASSET_DOMAINS[asset_apex])

    result["hotlinked_brands"] = sorted(hotlinked)

    # 2. Detect image-cloaked / canvas-only pages (minimal visible text + large embedded images/forms)
    body_text = soup.body.get_text(" ", strip=True) if soup.body else ""
    num_imgs = len(soup.find_all("img"))
    num_inputs = len(soup.find_all("input"))
    if len(body_text) < 90 and (num_imgs >= 1 or inline_data_imgs >= 1) and num_inputs >= 1:
        result["image_cloaked_suspected"] = True

    # 3. Fetch and fingerprint favicon bytes when live fetching is active
    if fetch_live:
        target_icon = urljoin(page_url, favicon_href) if favicon_href else urljoin(page_url, "/favicon.ico")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=3.5, verify=False) as client:
                icon_resp = await client.get(target_icon)
                if icon_resp.status_code == 200 and len(icon_resp.content) > 32:
                    result["favicon_sha256"] = hashlib.sha256(icon_resp.content).hexdigest()[:16]
                    final_icon_apex = get_apex_domain(str(icon_resp.url))
                    if final_icon_apex and final_icon_apex != page_apex and final_icon_apex in BRAND_ASSET_DOMAINS:
                        hotlinked.add(BRAND_ASSET_DOMAINS[final_icon_apex])
                        result["hotlinked_brands"] = sorted(hotlinked)
        except Exception:
            pass

    # Compute visual asset risk score
    v_risk = 0.0
    if result["hotlinked_brands"]:
        v_risk += 0.85
    if result["image_cloaked_suspected"]:
        v_risk += 0.70
    if inline_data_imgs >= 2 and len(body_text) < 120:
        v_risk += 0.45

    result["visual_risk_score"] = min(v_risk, 0.99)
    return result


def generate_explanations(
    url_feats: dict, dom_feats: dict, visual_info: dict,
    p_url: float, p_char: float, p_dom: float, p_txt: float,
    is_verified_brand: bool, html_available: int
) -> list[str]:
    if is_verified_brand:
        return [
            "Verified Official Brand Domain: HTTPS connection terminates on an authentic, verified apex domain.",
            "No Domain Spoofing Detected: Registered domain matches the protected brand identity."
        ]

    reasons = []
    if visual_info.get("hotlinked_brands"):
        brands_str = ", ".join(visual_info["hotlinked_brands"])
        reasons.append(f"Cross-Domain Brand Asset Hotlinking: Page loads logos/icons directly from official {brands_str} servers on an unverified domain.")
    if visual_info.get("image_cloaked_suspected"):
        reasons.append("Image-Cloaked Login Layout: Page embeds input fields alongside images with minimal readable text to evade text filters.")
    if url_feats.get("brand_mismatch"):
        reasons.append("Brand-Domain Mismatch: Protected brand name appears in subdomain/path, not the registered domain.")
    if p_char >= 0.75 and not url_feats.get("brand_mismatch"):
        reasons.append("Subword Typosquatting / Phishing Path Pattern: Character n-gram model flagged deceptive host/path structure.")
    if url_feats.get("has_ip_address"):
        reasons.append("Raw IP Address Host: URL uses a direct IP address instead of a registered domain name.")
    if url_feats.get("is_punycode"):
        reasons.append("Punycode / Homoglyph Domain: Internationalized domain encoding (xn--) detected.")
    if url_feats.get("suspicious_tld"):
        reasons.append("High-Risk Top-Level Domain (TLD): Domain uses a TLD frequently abused in phishing campaigns.")
    if not url_feats.get("is_https"):
        reasons.append("Unencrypted HTTP Connection: Page is not served over HTTPS.")
    if dom_feats.get("external_form_action"):
        reasons.append("External Credential Harvesting: HTML <form> submits user input to a different external domain.")
    if dom_feats.get("has_password_field") and (url_feats.get("brand_mismatch") or p_txt > 0.7 or visual_info.get("hotlinked_brands")):
        reasons.append("Credential Collection Form: Password input field detected on an unverified domain.")
    if dom_feats.get("js_obfuscation_hits", 0) > 0:
        reasons.append(f"Obfuscated JavaScript Detected: Found {dom_feats['js_obfuscation_hits']} dynamic decoding/eval calls.")
    if dom_feats.get("title_brand_mismatch"):
        reasons.append("Title Brand Spoofing: Page title impersonates a protected brand that does not own the domain.")
    if dom_feats.get("urgency_phrase_count", 0) > 0:
        reasons.append("Social Engineering Urgency Cues: Page text contains account suspension or verification threats.")
    if not html_available:
        reasons.append("Context Gate Active: Static HTML was unavailable, WAF-blocked, or an unhydrated JS shell; decision gated toward URL experts.")

    if not reasons:
        if max(p_url, p_char, p_dom, p_txt, visual_info.get("visual_risk_score", 0.0)) < 0.40:
            reasons.append("URL subwords, domain structure, DOM behavior, visual assets, and page text align with legitimate web patterns.")
        else:
            reasons.append("Subtle lexical, subword, and semantic HTML patterns contributed to the calibrated risk score.")
    return reasons


async def fetch_webpage_html(target_url: str) -> tuple[str, str, int]:
    normalized = target_url.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = "https://" + normalized

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0, verify=False) as client:
            resp = await client.get(normalized, headers=headers)
            if resp.status_code in {401, 403, 429, 503}:
                return str(resp.url), "", resp.status_code
            return str(resp.url), resp.text[:30000], resp.status_code
    except Exception:
        return normalized, "", 0


@app.post("/api/analyze")
async def analyze_url(req: AnalyzeRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    original_url = req.url.strip()
    if not original_url.startswith(("http://", "https://")):
        original_url = "https://" + original_url

    if is_local_or_private_host(original_url):
        return {
            "analyzed_url": original_url,
            "fetch_status": "local_network_skipped",
            "verdict": "LOCAL / PRIVATE",
            "calibrated_risk_score": 0.0,
            "expert_scores": {
                "url_lexical_risk": 0.0,
                "url_subword_risk": 0.0,
                "dom_behavior_risk": 0.0,
                "semantic_html_risk": 0.0,
                "visual_asset_risk": 0.0,
            },
            "visual_fingerprint": {},
            "explanations": [
                "Local Loopback / Private Network Host: Address resolves to a non-routable local environment rather than a public internet domain."
            ],
            "extracted_signals": {},
        }

    final_url = original_url
    html_content = req.html or ""
    fetch_status = "provided_html"
    http_status = 200

    if not html_content and req.fetch_live:
        final_url, html_content, http_status = await fetch_webpage_html(original_url)
        if http_status in {401, 403, 429, 503}:
            fetch_status = f"bot_blocked_http_{http_status}_url_gated"
        elif html_content:
            fetch_status = "live_fetched"
        else:
            fetch_status = "url_only_unreachable"

    is_verified_brand = check_verified_brand_domain(original_url) and check_verified_brand_domain(final_url)

    u_feats = extract_url_features(original_url)
    sub_str = normalize_url_for_subwords(original_url)
    d_feats = extract_dom_features(final_url, html_content)
    clean_txt = extract_clean_text(original_url, html_content)
    gate_feats = extract_modality_integrity(html_content, clean_txt, d_feats)
    visual_info = await analyze_visual_brand_assets(
        final_url, html_content, fetch_live=(req.fetch_live and not req.html), is_verified_brand=is_verified_brand
    )

    p_url = float(url_model.predict_proba(pd.DataFrame([u_feats]))[0, 1])
    p_char = float(char_model.predict_proba(char_tfidf.transform([sub_str]))[0, 1])
    p_dom = float(dom_model.predict_proba(pd.DataFrame([d_feats]))[0, 1]) if html_content else 0.50
    p_txt = float(txt_model.predict_proba(word_tfidf.transform([clean_txt]))[0, 1]) if html_content else 0.50
    p_vis = float(visual_info["visual_risk_score"])

    if is_verified_brand:
        fused_prob = min(p_url * 0.10, 0.02)
    else:
        meta_row = np.array([[
            p_url, p_char, p_dom, p_txt,
            gate_feats["html_available"],
            gate_feats["text_density_log"],
            gate_feats["has_any_form"],
        ]])
        base_fused = float(meta_model.predict_proba(meta_row)[0, 1])
        # Combine calibrated ensemble with visual brand-asset spoofing detector
        fused_prob = max(base_fused, p_vis)

    if fused_prob >= 0.70:
        verdict = "PHISHING"
    elif fused_prob >= 0.45:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"

    return {
        "analyzed_url": final_url,
        "fetch_status": fetch_status,
        "verdict": verdict,
        "calibrated_risk_score": round(fused_prob * 100, 2),
        "expert_scores": {
            "url_lexical_risk": round(p_url * 100, 2),
            "url_subword_risk": round(p_char * 100, 2),
            "dom_behavior_risk": round(p_dom * 100, 2),
            "semantic_html_risk": round(p_txt * 100, 2),
            "visual_asset_risk": round(p_vis * 100, 2),
        },
        "visual_fingerprint": {
            "favicon_sha256": visual_info["favicon_sha256"],
            "hotlinked_brands": visual_info["hotlinked_brands"],
            "image_cloaked_suspected": visual_info["image_cloaked_suspected"],
        },
        "explanations": generate_explanations(
            u_feats, d_feats, visual_info, p_url, p_char, p_dom, p_txt, is_verified_brand, gate_feats["html_available"]
        ),
        "extracted_signals": {**u_feats, **d_feats, **gate_feats},
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PhishGuard 2.1 | 5-Signal Multimodal Phishing Detector</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <div class="max-w-5xl mx-auto px-6 py-12">
    <div class="mb-8">
      <span class="text-xs uppercase tracking-widest text-cyan-400 font-semibold">4-Expert Context-Gated Ensemble + Visual Asset Integrity</span>
      <h1 class="text-3xl font-bold mt-1">PhishGuard Threat Analyzer</h1>
      <p class="text-slate-400 text-sm mt-1">Evaluates URL lexical structure, subword n-grams, DOM behavior, HTML semantics, and favicon/brand asset hotlinking.</p>
    </div>

    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <label class="block text-sm font-medium text-slate-300 mb-2">Target URL</label>
      <div class="flex gap-3">
        <input id="urlInput" type="text" placeholder="https://example.com/login"
          class="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-cyan-400" />
        <button onclick="runScan()" id="scanBtn"
          class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold px-6 py-3 rounded-xl text-sm transition">
          Analyze URL
        </button>
      </div>

      <div class="mt-4 flex flex-wrap gap-2 text-xs">
        <span class="text-slate-500 py-1">Quick Demos:</span>
        <button onclick="setDemo('https://gemini.google.com/app', '')" class="bg-slate-800 hover:bg-slate-700 px-3 py-1 rounded-lg text-emerald-300">Gemini Web App (Safe)</button>
        <button onclick="setDemo('https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster', '')" class="bg-slate-800 hover:bg-slate-700 px-3 py-1 rounded-lg text-emerald-300">StackOverflow Thread (WAF-Gated)</button>
        <button onclick="setDemo('http://127.0.0.1:8000/', '')" class="bg-slate-800 hover:bg-slate-700 px-3 py-1 rounded-lg text-slate-300">Localhost Dev Server (Private)</button>
        <button onclick="setDemo('https://secure-auth-portal.xyz/signin', '<html><head><link rel=\\'icon\\' href=\\'https://www.paypalobjects.com/favicon.ico\\'></head><body><img src=\\'https://www.paypalobjects.com/logo.png\\'><form><input type=\\'password\\'></form></body></html>')" class="bg-slate-800 hover:bg-slate-700 px-3 py-1 rounded-lg text-rose-300">Cloaked Brand-Asset Spoof (Phish)</button>
      </div>

      <details class="mt-4">
        <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200">Optional: Paste Raw HTML Payload (for offline / cloaked simulation)</summary>
        <textarea id="htmlInput" rows="4" placeholder="<html>...</html>"
          class="mt-2 w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-300"></textarea>
      </details>
    </div>

    <div id="resultCard" class="hidden mt-6 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div class="flex items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <div class="text-xs text-slate-400">Calibrated Phishing Probability</div>
          <div id="riskScore" class="text-4xl font-extrabold mt-1">0.0%</div>
          <div id="faviconMeta" class="text-xs text-slate-500 mt-1"></div>
        </div>
        <span id="verdictBadge" class="px-4 py-2 rounded-full text-sm font-bold tracking-wide">SAFE</span>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-5 gap-3 my-6">
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">URL Lexical</div>
          <div id="urlScore" class="text-xl font-bold mt-1 text-cyan-300">0%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">URL Subword</div>
          <div id="charScore" class="text-xl font-bold mt-1 text-sky-300">0%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">DOM Behavior</div>
          <div id="domScore" class="text-xl font-bold mt-1 text-purple-300">0%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Semantic HTML</div>
          <div id="txtScore" class="text-xl font-bold mt-1 text-amber-300">0%</div>
        </div>
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Visual Assets</div>
          <div id="visScore" class="text-xl font-bold mt-1 text-rose-300">0%</div>
        </div>
      </div>

      <div>
        <h3 class="text-sm font-semibold text-slate-300 mb-2">Explainable AI (XAI) Key Risk Indicators</h3>
        <ul id="reasonsList" class="space-y-2 text-sm text-slate-300 list-disc list-inside"></ul>
      </div>
    </div>
  </div>

  <script>
    function setDemo(url, html) {
      document.getElementById('urlInput').value = url;
      document.getElementById('htmlInput').value = html;
      runScan();
    }

    async function runScan() {
      const url = document.getElementById('urlInput').value.trim();
      const html = document.getElementById('htmlInput').value.trim();
      if (!url) return;

      const btn = document.getElementById('scanBtn');
      btn.textContent = 'Analyzing...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, html: html || null, fetch_live: !html })
        });
        const data = await res.json();

        document.getElementById('resultCard').classList.remove('hidden');
        document.getElementById('riskScore').textContent = data.calibrated_risk_score + '%';
        document.getElementById('urlScore').textContent = data.expert_scores.url_lexical_risk + '%';
        document.getElementById('charScore').textContent = data.expert_scores.url_subword_risk + '%';
        document.getElementById('domScore').textContent = data.expert_scores.dom_behavior_risk + '%';
        document.getElementById('txtScore').textContent = data.expert_scores.semantic_html_risk + '%';
        document.getElementById('visScore').textContent = data.expert_scores.visual_asset_risk + '%';

        const fav = data.visual_fingerprint?.favicon_sha256;
        document.getElementById('faviconMeta').textContent = fav ? `Favicon SHA-256: ${fav}` : '';

        const badge = document.getElementById('verdictBadge');
        badge.textContent = data.verdict;
        badge.className = 'px-4 py-2 rounded-full text-sm font-bold tracking-wide ' + (
          data.verdict === 'PHISHING' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' :
          data.verdict === 'SUSPICIOUS' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :
          data.verdict === 'LOCAL / PRIVATE' ? 'bg-slate-700/60 text-slate-300 border border-slate-600' :
          'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
        );

        const list = document.getElementById('reasonsList');
        list.innerHTML = data.explanations.map(r => `<li>${r}</li>`).join('');
      } catch (err) {
        alert('Error analyzing URL: ' + err);
      } finally {
        btn.textContent = 'Analyze URL';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""