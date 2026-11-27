import importlib
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Load features (or fallback to features_3 if named features_3.py)
try:
    feat_mod = importlib.import_module("features")
except ModuleNotFoundError:
    feat_mod = importlib.import_module("features_3")

extract_url_features = feat_mod.extract_url_features
extract_dom_features = feat_mod.extract_dom_features
extract_clean_text = feat_mod.extract_clean_text
normalize_url_for_subwords = feat_mod.normalize_url_for_subwords
extract_modality_integrity = feat_mod.extract_modality_integrity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("PHRESHPHISH_DATA_DIR", ROOT_DIR / "data" / "processed"))
TRAIN_PATH = Path(os.getenv("PHRESHPHISH_TRAIN_PATH", DATA_DIR / "train.parquet"))
TEST_PATH = Path(os.getenv("PHRESHPHISH_TEST_PATH", DATA_DIR / "test.parquet"))

OUTPUT_DIR = Path(os.getenv("PHRESHPHISH_OUT_DIR", ROOT_DIR / "artifacts" / "v3_gated_4expert"))
BUNDLE_PATH = Path(
    os.getenv(
        "PHRESHPHISH_BUNDLE_PATH",
        ROOT_DIR / "models" / "phishguard_bundle.joblib",
    )
)
BUNDLE_VERSION = os.getenv("PHRESHPHISH_BUNDLE_VERSION", "3.0-25k-gated-4expert")

RANDOM_STATE = 42
N_SPLITS = 5

WORD_TFIDF_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}

CHAR_TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "max_features": 30000,
    "sublinear_tf": True,
}


def pos_proba(estimator, X) -> np.ndarray:
    """Extract probability for positive class (1)."""
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=np.float64)
    return np.asarray(proba[:, -1], dtype=np.float64)


def _normalize_shard(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = {"url", "html", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    out = df.copy().reset_index(drop=True)
    out["url"] = out["url"].fillna("").astype(str)
    out["html"] = out["html"].fillna("").astype(str)
    out["label"] = out["label"].astype(int)
    return out


def load_train_test_shards(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split local train and test Parquet shards."""
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected local Parquet shards at '{train_path}' and '{test_path}'."
        )
    train_df = _normalize_shard(pd.read_parquet(train_path), str(train_path))
    test_df = _normalize_shard(pd.read_parquet(test_path), str(test_path))
    return train_df, test_df


def build_feature_matrices(df: pd.DataFrame):
    """Build URL, DOM, clean text, normalized subword URLs, and 3-feature modality gating."""
    url_rows = []
    dom_rows = []
    clean_texts = []
    subword_urls = []
    mod_rows = []

    for row in df.itertuples(index=False):
        url = row.url
        html = row.html

        u_feats = extract_url_features(url)
        d_feats = extract_dom_features(url, html)
        c_text = extract_clean_text(url, html)
        sub_url = normalize_url_for_subwords(url)
        m_feats = extract_modality_integrity(html, c_text, d_feats)

        url_rows.append(u_feats)
        dom_rows.append(d_feats)
        clean_texts.append(c_text)
        subword_urls.append(sub_url)
        mod_rows.append(m_feats)

    X_url = pd.DataFrame(url_rows).fillna(0.0)
    X_dom = pd.DataFrame(dom_rows).fillna(0.0)
    X_mod_df = pd.DataFrame(mod_rows).fillna(0.0)
    mod_cols = list(X_mod_df.columns)
    X_mod = X_mod_df.to_numpy(dtype=np.float64)

    return X_url, X_dom, clean_texts, subword_urls, X_mod, mod_cols


def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_linear() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


def make_meta() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_evaluations(
    y_test: np.ndarray,
    test_probs: dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    url_model: LGBMClassifier,
    url_feature_names: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ablation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = list(metrics_df.index)
    axes[0].barh(order, metrics_df["roc_auc"])
    axes[0].set_title("Test ROC-AUC by Expert / Gated Meta")
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].invert_yaxis()

    axes[1].barh(order, metrics_df["f1"])
    axes[1].set_title("Test F1 Score (@0.50) by Expert / Gated Meta")
    axes[1].set_xlabel("F1 Score")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_dir / "01_ablation_comparison.jpg", dpi=150)
    plt.close(fig)

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("Test ROC Curves (4-Expert + Gated Meta)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "02_roc_curves.jpg", dpi=150)
    plt.close(fig)

    # 3. Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in test_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_title("Test Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "03_calibration_curve.png", dpi=150)
    plt.close(fig)

    # 4. Feature Importance / SHAP Summary on URL Expert
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(url_model)
        shap_values = explainer.shap_values(
            np.zeros((min(200, len(url_feature_names)), len(url_feature_names)))
        )
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(vals).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1]
        ax.barh([url_feature_names[i] for i in idx], mean_abs[idx])
        ax.set_title("URL Expert Mean |SHAP| Importance")
        ax.set_xlabel("Mean |SHAP value|")
        ax.invert_yaxis()
    except Exception:
        importances = np.asarray(url_model.feature_importances_, dtype=float)
        idx = np.argsort(importances)[::-1]
        ax.barh([url_feature_names[i] for i in idx], importances[idx])
        ax.set_title("URL Expert Split Importance")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_dir / "04_shap_summary.jpg", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading local shards from '{TRAIN_PATH}' and '{TEST_PATH}'...")
    train_df, test_df = load_train_test_shards(TRAIN_PATH, TEST_PATH)
    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    X_url_tr, X_dom_tr, txt_tr, sub_tr, mod_tr, mod_cols = build_feature_matrices(train_df)
    X_url_te, X_dom_te, txt_te, sub_te, mod_te, _ = build_feature_matrices(test_df)

    y_tr = train_df["label"].to_numpy(dtype=int)
    y_te = test_df["label"].to_numpy(dtype=int)

    print(f"[2/5] Running {N_SPLITS}-fold Stratified CV across 4 experts on train shard...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_url = np.zeros(len(train_df), dtype=np.float64)
    oof_char = np.zeros(len(train_df), dtype=np.float64)
    oof_dom = np.zeros(len(train_df), dtype=np.float64)
    oof_txt = np.zeros(len(train_df), dtype=np.float64)

    for fold, (f_tr, f_va) in enumerate(skf.split(X_url_tr, y_tr), start=1):
        m_url = make_lgbm().fit(X_url_tr.iloc[f_tr], y_tr[f_tr])
        oof_url[f_va] = pos_proba(m_url, X_url_tr.iloc[f_va])

        fold_char_vec = TfidfVectorizer(**CHAR_TFIDF_PARAMS)
        X_c_tr = fold_char_vec.fit_transform([sub_tr[i] for i in f_tr])
        X_c_va = fold_char_vec.transform([sub_tr[i] for i in f_va])
        m_char = make_linear().fit(X_c_tr, y_tr[f_tr])
        oof_char[f_va] = pos_proba(m_char, X_c_va)

        m_dom = make_lgbm().fit(X_dom_tr.iloc[f_tr], y_tr[f_tr])
        oof_dom[f_va] = pos_proba(m_dom, X_dom_tr.iloc[f_va])

        fold_word_vec = TfidfVectorizer(**WORD_TFIDF_PARAMS)
        X_w_tr = fold_word_vec.fit_transform([txt_tr[i] for i in f_tr])
        X_w_va = fold_word_vec.transform([txt_tr[i] for i in f_va])
        m_txt = make_linear().fit(X_w_tr, y_tr[f_tr])
        oof_txt[f_va] = pos_proba(m_txt, X_w_va)

        print(
            f"  Fold {fold}/{N_SPLITS} | "
            f"URL: {roc_auc_score(y_tr[f_va], oof_url[f_va]):.4f} | "
            f"CHAR: {roc_auc_score(y_tr[f_va], oof_char[f_va]):.4f} | "
            f"DOM: {roc_auc_score(y_tr[f_va], oof_dom[f_va]):.4f} | "
            f"TXT: {roc_auc_score(y_tr[f_va], oof_txt[f_va]):.4f}"
        )

    print("[3/5] Fitting 7-feature gated meta_model on OOF probabilities + modality integrity...")
    # Canonical ordering: [p_url, p_char, p_dom, p_txt, html_available, text_density_log, has_any_form]
    X_meta_oof = np.column_stack([oof_url, oof_char, oof_dom, oof_txt, mod_tr])
    meta_model = make_meta().fit(X_meta_oof, y_tr)
    print(f"  OOF Gated Meta ROC-AUC: {roc_auc_score(y_tr, pos_proba(meta_model, X_meta_oof)):.4f}")

    print("[4/5] Refitting full-train base experts & vectorizers and evaluating on held-out test shard...")
    url_model = make_lgbm().fit(X_url_tr, y_tr)

    char_tfidf = TfidfVectorizer(**CHAR_TFIDF_PARAMS).fit(sub_tr)
    char_model = make_linear().fit(char_tfidf.transform(sub_tr), y_tr)

    dom_model = make_lgbm().fit(X_dom_tr, y_tr)

    tfidf = TfidfVectorizer(**WORD_TFIDF_PARAMS).fit(txt_tr)
    txt_model = make_linear().fit(tfidf.transform(txt_tr), y_tr)

    p_url_te = pos_proba(url_model, X_url_te)
    p_char_te = pos_proba(char_model, char_tfidf.transform(sub_te))
    p_dom_te = pos_proba(dom_model, X_dom_te)
    p_txt_te = pos_proba(txt_model, tfidf.transform(txt_te))

    X_meta_te = np.column_stack([p_url_te, p_char_te, p_dom_te, p_txt_te, mod_te])
    p_gated_te = pos_proba(meta_model, X_meta_te)

    test_probs = {
        "URL Expert": p_url_te,
        "Char Subword Expert": p_char_te,
        "DOM Expert": p_dom_te,
        "HTML Text Expert": p_txt_te,
        "Gated 4-Expert Meta": p_gated_te,
    }

    rows = {name: compute_metrics(y_te, probs) for name, probs in test_probs.items()}
    metrics_df = pd.DataFrame.from_dict(rows, orient="index")
    print("\n--- Held-Out Test Metrics ---")
    print(metrics_df.round(4).to_string())

    y_pred_te = (p_gated_te >= 0.5).astype(int)
    print("\n--- Gated 4-Expert Meta Classification Report (@0.50) ---")
    print(classification_report(y_te, y_pred_te, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred_te))

    print("[5/5] Saving evaluation plots and model bundle...")
    plot_evaluations(
        y_test=y_te,
        test_probs=test_probs,
        metrics_df=metrics_df,
        url_model=url_model,
        url_feature_names=list(X_url_tr.columns),
        out_dir=OUTPUT_DIR,
    )
    metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv")

    bundle = {
        "version": BUNDLE_VERSION,
        "url_model": url_model,
        "char_model": char_model,
        "dom_model": dom_model,
        "txt_model": txt_model,
        "char_tfidf": char_tfidf,
        "tfidf": tfidf,
        "meta_model": meta_model,
        "url_features": list(X_url_tr.columns),
        "dom_features": list(X_dom_tr.columns),
        "modality_features": mod_cols,
        "meta_features": ["p_url", "p_char", "p_dom", "p_txt", *mod_cols],
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved bundle to {BUNDLE_PATH}")


if __name__ == "__main__":
    main()