import asyncio
import importlib
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tldextract
from bs4 import BeautifulSoup
from huggingface_hub import HfFileSystem
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    feat_mod = importlib.import_module("features_3")
except ModuleNotFoundError:
    feat_mod = importlib.import_module("features")

extract_clean_text = feat_mod.extract_clean_text
extract_dom_features = feat_mod.extract_dom_features
extract_modality_integrity = feat_mod.extract_modality_integrity
extract_url_features = feat_mod.extract_url_features
normalize_url_for_subwords = feat_mod.normalize_url_for_subwords


def resolve_bundle_path() -> Path:
    env_path = os.getenv("PHRESHPHISH_BUNDLE_PATH")
    if env_path:
        return Path(env_path)
    v3_default = ROOT_DIR / "artifacts" / "v3_gated_4expert" / "phreshphish_3.0-25k-gated-4expert.joblib"
    legacy_default = ROOT_DIR / "models" / "phishguard_bundle.joblib"
    return v3_default if v3_default.exists() else legacy_default


def pos_proba_single(model, X) -> float:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else -1
    return float(proba[0, idx])


BUNDLE_PATH = resolve_bundle_path()
bundle = joblib.load(BUNDLE_PATH)
url_model = bundle["url_model"]
char_model = bundle["char_model"]
char_tfidf = bundle["char_tfidf"]
dom_model = bundle["dom_model"]
txt_model = bundle["txt_model"]
word_tfidf = bundle["tfidf"]
meta_model = bundle["meta_model"]

URL_FEATURE_COLS = bundle.get("url_features")
DOM_FEATURE_COLS = bundle.get("dom_features")

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
    "gstatic.com": "Google",
    "googleapis.com": "Google",
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


def get_apex(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def compute_visual_asset_risk(page_url: str, html: str) -> tuple[float, list[str]]:
    if not html:
        return 0.0, []
    page_apex = get_apex(page_url)
    soup = BeautifulSoup(html, "lxml")

    asset_urls = []
    for link in soup.find_all("link", href=True):
        rels = [r.lower() for r in (link.get("rel") or [])]
        if any("icon" in r for r in rels):
            asset_urls.append(link["href"].strip())

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

    hotlinked = set()
    for raw_asset in asset_urls:
        if raw_asset.startswith("data:"):
            continue
        abs_asset = urljoin(page_url, raw_asset)
        asset_apex = get_apex(abs_asset)
        if asset_apex and asset_apex != page_apex and asset_apex in BRAND_ASSET_DOMAINS:
            hotlinked.add(BRAND_ASSET_DOMAINS[asset_apex])

    body_text = soup.body.get_text(" ", strip=True) if soup.body else ""
    num_imgs = len(soup.find_all("img"))
    num_inputs = len(soup.find_all("input"))
    image_cloaked = len(body_text) < 90 and (num_imgs >= 1 or inline_data_imgs >= 1) and num_inputs >= 1

    v_risk = 0.0
    if hotlinked:
        v_risk += 0.85
    if image_cloaked:
        v_risk += 0.70
    if inline_data_imgs >= 2 and len(body_text) < 120:
        v_risk += 0.45

    return min(v_risk, 0.99), sorted(hotlinked)


def predict_sample_v21(url: str, html: str) -> dict:
    u_feats = extract_url_features(url)
    sub_str = normalize_url_for_subwords(url)
    d_feats = extract_dom_features(url, html)
    clean_txt = extract_clean_text(url, html)
    gate_feats = extract_modality_integrity(html, clean_txt, d_feats)
    p_vis, hotlinked = compute_visual_asset_risk(url, html)

    url_df = pd.DataFrame([{k: u_feats.get(k, 0.0) for k in URL_FEATURE_COLS}] if URL_FEATURE_COLS else [u_feats]).fillna(0.0)
    dom_df = pd.DataFrame([{k: d_feats.get(k, 0.0) for k in DOM_FEATURE_COLS}] if DOM_FEATURE_COLS else [d_feats]).fillna(0.0)

    p_url = pos_proba_single(url_model, url_df)
    p_char = pos_proba_single(char_model, char_tfidf.transform([sub_str]))
    p_dom = pos_proba_single(dom_model, dom_df) if html else 0.50
    p_txt = pos_proba_single(txt_model, word_tfidf.transform([clean_txt])) if html else 0.50

    # Canonical 7-feature meta row aligned with 02_train_and_evaluate.py:
    # [p_url, p_char, p_dom, p_txt, html_available, text_density_log, has_any_form]
    meta_row = np.array([[
        p_url,
        p_char,
        p_dom,
        p_txt,
        float(gate_feats["html_available"]),
        float(gate_feats["text_density_log"]),
        float(gate_feats["has_any_form"]),
    ]], dtype=np.float64)

    base_fused = pos_proba_single(meta_model, meta_row)
    p_fused = max(base_fused, p_vis)

    return {
        "p_url": p_url,
        "p_char": p_char,
        "p_dom": p_dom,
        "p_txt": p_txt,
        "p_vis": p_vis,
        "p_fused": p_fused,
        "html_gated": not bool(gate_feats["html_available"]),
        "hotlinked": hotlinked,
    }


def run_domain_disjoint_eval(target_per_class: int = 500):
    print(f"\n=== PART A: DOMAIN-DISJOINT UNSEEN SHARD EVALUATION (Bundle: {BUNDLE_PATH.name}) ===")
    train_df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "train.parquet")
    test_df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "test.parquet")
    seen_domains = set(train_df["url"].map(get_apex)) | set(test_df["url"].map(get_apex))
    print(f"Excluding {len(seen_domains)} previously seen apex domains...")

    fs = HfFileSystem()
    files = fs.glob("datasets/phreshphish/phreshphish/**/*.parquet")

    unseen_phish, unseen_benign = [], []
    for file_path in files[15:]:
        with fs.open(file_path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=500, columns=["url", "html", "label"]):
                for _, row in batch.to_pandas().iterrows():
                    url = str(row.get("url") or "").strip()
                    html = str(row.get("html") or "").strip()
                    label_raw = str(row.get("label") or "").strip().lower()
                    if not url or len(html) < 150:
                        continue
                    apex = get_apex(url)
                    if not apex or apex in seen_domains:
                        continue
                    seen_domains.add(apex)

                    lbl = 1 if label_raw in {"phish", "1", "true"} else 0
                    if lbl == 1 and len(unseen_phish) < target_per_class:
                        unseen_phish.append((url, html[:30000], 1))
                    elif lbl == 0 and len(unseen_benign) < target_per_class:
                        unseen_benign.append((url, html[:30000], 0))

                    if len(unseen_phish) >= target_per_class and len(unseen_benign) >= target_per_class:
                        break
                if len(unseen_phish) >= target_per_class and len(unseen_benign) >= target_per_class:
                    break
        if len(unseen_phish) >= target_per_class and len(unseen_benign) >= target_per_class:
            break

    samples = unseen_phish + unseen_benign
    y_true = np.array([s[2] for s in samples])
    preds = [predict_sample_v21(u, h) for u, h, _ in samples]
    y_prob = np.array([p["p_fused"] for p in preds])
    y_bin = (y_prob >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    print(f"Evaluated {len(samples)} strictly unseen domains ({tp + fn} phish, {tn + fp} benign):")
    print(f"  ROC-AUC:    {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  F1-Score:   {f1_score(y_true, y_bin):.4f}")
    print(f"  Precision:  {precision_score(y_true, y_bin):.4f}")
    print(f"  Recall:     {recall_score(y_true, y_bin):.4f}")
    print(f"  Brier Loss: {brier_score_loss(y_true, y_prob):.4f}")
    print(f"  False Positive Rate: {fp / max(tn + fp, 1):.2%} ({fp}/{tn + fp})")
    print(f"  False Negative Rate: {fn / max(tp + fn, 1):.2%} ({fn}/{tp + fn})")

    fp_cases = [
        (samples[i][0], preds[i])
        for i in range(len(samples))
        if y_true[i] == 0 and y_bin[i] == 1
    ]
    fp_cases.sort(key=lambda x: x[1]["p_fused"], reverse=True)
    print("\nTop 5 False Positives on Unseen Benign Domains:")
    for u, p in fp_cases[:5]:
        print(
            f"  {u[:75]} -> Fused={p['p_fused']:.2%} "
            f"(URL={p['p_url']:.2%}, Sub={p['p_char']:.2%}, DOM={p['p_dom']:.2%}, Txt={p['p_txt']:.2%}, Vis={p['p_vis']:.2%})"
        )


async def run_live_web_audit():
    print("\n=== PART B: LIVE REAL-WORLD UNSEEN WEBSITE AUDIT ===")
    live_benign_urls = [
        "https://docs.python.org/3/library/asyncio.html",
        "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
        "https://www.wikipedia.org/",
        "https://www.mit.edu/",
        "https://www.incometax.gov.in/iec/foportal/",
        "https://www.irctc.co.in/nget/train-search",
        "https://huggingface.co/datasets/phreshphish/phreshphish",
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "https://www.cloudflare.com/",
        "https://www.notion.so/",
        "https://www.canva.com/",
        "https://www.coursera.org/",
        "https://www.nytimes.com/",
        "https://www.bbc.com/news",
        "https://www.reddit.com/r/MachineLearning/",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=8.0, verify=False) as client:
        for url in live_benign_urls:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code in {401, 403, 429, 503}:
                    html = ""
                    status = f"HTTP {resp.status_code} (Gated)"
                else:
                    html = resp.text[:30000]
                    status = f"HTTP {resp.status_code}"
            except Exception as e:
                html = ""
                status = f"ERR ({type(e).__name__})"

            res = predict_sample_v21(url, html)
            flag = "FAIL (FP)" if res["p_fused"] >= 0.50 else "PASS"
            print(
                f"[{flag:9s}] {url:<55} | {status:<18} | "
                f"Fused={res['p_fused']:6.2%} (URL={res['p_url']:5.2%}, Sub={res['p_char']:5.2%}, DOM={res['p_dom']:5.2%}, Txt={res['p_txt']:5.2%}, Vis={res['p_vis']:5.2%})"
            )


if __name__ == "__main__":
    run_domain_disjoint_eval(target_per_class=500)
    asyncio.run(run_live_web_audit())