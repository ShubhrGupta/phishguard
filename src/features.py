import math
import re
from urllib.parse import urlparse
import tldextract
from bs4 import BeautifulSoup

PROTECTED_BRANDS = {
    "google", "gmail", "microsoft", "office", "outlook", "apple", "icloud",
    "paypal", "amazon", "netflix", "facebook", "instagram", "linkedin",
    "chase", "wellsfargo", "bankofamerica", "binance", "coinbase", "adobe",
    "dropbox", "dhl", "fedex", "usps", "sbi", "hdfc", "icici"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "tk", "ml", "ga", "cf", "gq",
    "work", "click", "link", "live", "support", "vip", "icu", "buzz"
}

URGENCY_PHRASES = [
    "verify your account", "account suspended", "unauthorized login",
    "confirm your identity", "update payment", "unusual activity",
    "security alert", "limited time", "action required", "validate your"
]

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

def extract_url_features(url: str) -> dict:
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")
    ext = tldextract.extract(clean_url)

    reg_domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    full_domain = f"{reg_domain}.{suffix}" if suffix else reg_domain
    subdomain = ext.subdomain.lower()
    path_query = f"{parsed.path}?{parsed.query}".lower()

    # Brand-Domain Mismatch: brand name in subdomain/path when registered domain is NOT that brand
    brand_in_sub_or_path = any(b in subdomain or b in path_query for b in PROTECTED_BRANDS)
    brand_is_owner = any(b == reg_domain for b in PROTECTED_BRANDS)
    norm_host = normalize_homoglyphs(f"{ext.subdomain}.{ext.domain}")
    norm_apex = normalize_homoglyphs(ext.domain)

    brand_mismatch = int(
        any(
            (b in f"{ext.subdomain}.{ext.domain}.{ext.suffix}".lower() or b in norm_host)
            and (ext.domain.lower() != b and norm_apex != b)
            for b in PROTECTED_BRANDS
        )
    )

    # Count digits & hyphens strictly in the domain/subdomain (not deep legitimate paths)
    host_str = f"{subdomain}.{full_domain}" if subdomain else full_domain
    digit_ratio_host = sum(c.isdigit() for c in host_str) / max(len(host_str), 1)

    return {
        "is_https": int(parsed.scheme == "https"),
        "has_ip_address": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", reg_domain))),
        "is_punycode": int("xn--" in host_str),
        "domain_length": len(full_domain),
        "domain_entropy": shannon_entropy(reg_domain),
        "subdomain_count": len(subdomain.split(".")) if subdomain else 0,
        "has_hyphen_in_domain": int("-" in reg_domain),
        "digit_ratio_host": digit_ratio_host,
        "at_symbol_in_url": int("@" in clean_url),
        "brand_mismatch": brand_mismatch,
        "suspicious_tld": int(suffix in SUSPICIOUS_TLDS),
        "non_standard_port": int(parsed.port not in (None, 80, 443)),
    }

def extract_dom_features(url: str, html: str) -> dict:
    soup = BeautifulSoup(html or "", "lxml")
    ext = tldextract.extract(url)
    page_domain = f"{ext.domain}.{ext.suffix}".lower()

    forms = soup.find_all("form")
    password_inputs = soup.find_all("input", {"type": "password"})
    email_inputs = soup.find_all("input", {"type": "email"})
    hidden_inputs = soup.find_all("input", {"type": "hidden"})

    external_form_action = 0
    empty_or_js_form_action = 0
    for form in forms:
        action = (form.get("action") or "").strip().lower()
        if not action or action.startswith("#") or action.startswith("javascript:"):
            empty_or_js_form_action = 1
        elif action.startswith("http"):
            act_ext = tldextract.extract(action)
            act_domain = f"{act_ext.domain}.{act_ext.suffix}".lower()
            if act_domain and act_domain != page_domain:
                external_form_action = 1

    # External scripts & links
    scripts = [s.get("src", "") for s in soup.find_all("script", src=True)]
    ext_scripts = sum(1 for s in scripts if s.startswith("http") and page_domain not in s)

    links = [a.get("href", "") for a in soup.find_all("a", href=True)]
    null_links = sum(1 for l in links if l in ("", "#", "#!", "javascript:void(0)", "javascript:;"))

    # Obfuscated JavaScript patterns
    raw_scripts_text = " ".join(s.get_text() for s in soup.find_all("script") if not s.get("src"))
    obfuscation_hits = sum(
        raw_scripts_text.count(tok)
        for tok in ["eval(", "atob(", "fromCharCode", "unescape(", "document.write("]
    )

    # Check if title mentions a protected brand that doesn't match page_domain
    title_text = (soup.title.string if soup.title and soup.title.string else "").lower()
    title_brand_mismatch = int(
        any(b in title_text for b in PROTECTED_BRANDS)
        and not any(b == ext.domain.lower() for b in PROTECTED_BRANDS)
    )

    visible_text = soup.get_text(" ", strip=True).lower()
    urgency_count = sum(phrase in visible_text for phrase in URGENCY_PHRASES)

    return {
        "num_forms": len(forms),
        "has_password_field": int(len(password_inputs) > 0),
        "has_email_field": int(len(email_inputs) > 0),
        "hidden_input_count": len(hidden_inputs),
        "external_form_action": external_form_action,
        "empty_or_js_form_action": empty_or_js_form_action,
        "external_script_ratio": ext_scripts / max(len(scripts), 1),
        "null_link_ratio": null_links / max(len(links), 1),
        "js_obfuscation_hits": min(obfuscation_hits, 20),
        "title_brand_mismatch": title_brand_mismatch,
        "urgency_phrase_count": urgency_count,
        "iframe_count": len(soup.find_all("iframe")),
    }

def extract_clean_text(url: str, html: str) -> str:
    """Sanitizes HTML (removes hidden anti-analysis elements) and builds a structured document for TF-IDF."""
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Strip hidden elements (defends against indirect text injection)
    for hidden in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.I)):
        hidden.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    forms_info = " ".join(f"form_action:{f.get('action', '')}" for f in soup.find_all("form"))
    body_text = soup.get_text(" ", strip=True)[:4000]
    return f"URL: {url} | TITLE: {title} | {forms_info} | TEXT: {body_text}"

def normalize_url_for_subwords(url: str) -> str:
    """Extracts scheme + host + normalized path (strips volatile query tokens) for character n-gram modeling."""
    clean = url.strip().lower()
    parsed = urlparse(clean if "://" in clean else f"http://{clean}")
    host = parsed.netloc
    path = re.sub(r"\d+", "0", parsed.path or "/")
    return f"{host}{path}"


def extract_modality_integrity(html: str, clean_txt: str, dom_feats: dict) -> dict:
    """Quantifies HTML/Text signal availability so the meta-classifier can dynamically gate unreliable modalities."""
    raw_len = len(html or "")
    stripped_body = clean_txt.split("TEXT:", 1)[-1].strip() if "TEXT:" in clean_txt else ""
    body_len = len(stripped_body)
    is_blocked = int(
        raw_len == 0
        or body_len < 100
        or "just a moment" in stripped_body.lower()
        or "access denied" in stripped_body.lower()
    )
    return {
        "html_available": int(raw_len > 150 and not is_blocked),
        "text_density_log": math.log1p(body_len),
        "has_any_form": int(dom_feats.get("num_forms", 0) > 0),
    }
HOMOGLYPH_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})

MULTI_CHAR_GLYPHS = (
    ("rn", "m"),
    ("vv", "w"),
    ("cl", "d"),
)

def normalize_homoglyphs(text: str) -> str:
    """Converts common visual character substitutions (e.g., paypa1 -> paypal, rnicrosoft -> microsoft)."""
    s = text.lower().translate(HOMOGLYPH_MAP)
    for fake, real in MULTI_CHAR_GLYPHS:
        s = s.replace(fake, real)
    return s