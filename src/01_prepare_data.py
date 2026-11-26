import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from sklearn.model_selection import train_test_split

print("Connecting to PhreshPhish Parquet shards on Hugging Face...")
fs = HfFileSystem()
files = fs.glob("datasets/phreshphish/phreshphish/**/*.parquet")
print(f"Found {len(files)} parquet shards. Streaming rows...")

phish_rows, benign_rows = [], []
TARGET_PER_CLASS = 3000

for file_path in files:
    with fs.open(file_path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=500, columns=["url", "html", "label"]):
            df_batch = batch.to_pandas()
            for _, row in df_batch.iterrows():
                url = str(row.get("url") or "").strip()
                html = str(row.get("html") or "").strip()
                label = str(row.get("label") or "").strip().lower()

                if not url or len(html) < 150:
                    continue

                record = {
                    "url": url,
                    "html": html[:30000],
                    "label": 1 if label in {"phish", "1", "true"} else 0,
                }

                if record["label"] == 1 and len(phish_rows) < TARGET_PER_CLASS:
                    phish_rows.append(record)
                elif record["label"] == 0 and len(benign_rows) < TARGET_PER_CLASS:
                    benign_rows.append(record)

                if len(phish_rows) >= TARGET_PER_CLASS and len(benign_rows) >= TARGET_PER_CLASS:
                    break
            if len(phish_rows) >= TARGET_PER_CLASS and len(benign_rows) >= TARGET_PER_CLASS:
                break
    if len(phish_rows) >= TARGET_PER_CLASS and len(benign_rows) >= TARGET_PER_CLASS:
        break

df = pd.DataFrame(phish_rows + benign_rows)

# Hard-negative injection: deep-path legitimate web apps & OAuth flows
hard_negatives = [
    {"url": "https://mail.google.com/mail/u/0/#inbox", "html": "<html><head><title>Gmail</title></head><body><form action='https://accounts.google.com/signin'></form></body></html>", "label": 0},
    {"url": "https://accounts.google.com/v3/signin/identifier?continue=https://mail.google.com&flowName=GlifWebSignIn", "html": "<html><head><title>Sign in - Google Accounts</title></head><body>Google Account Sign In</body></html>", "label": 0},
    {"url": "https://github.com/torvalds/linux/blob/master/Makefile", "html": "<html><head><title>linux/Makefile at master · torvalds/linux · GitHub</title></head><body>Repository file view</body></html>", "label": 0},
    {"url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=abc&redirect_uri=https://portal.azure.com", "html": "<html><head><title>Sign in to your account</title></head><body>Microsoft Azure Active Directory</body></html>", "label": 0},
    {"url": "https://www.amazon.in/gp/css/order-history?ref_=nav_orders_first", "html": "<html><head><title>Your Orders</title></head><body>Amazon Order History</body></html>", "label": 0},
] * 30

df = pd.concat([df, pd.DataFrame(hard_negatives)], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
train_df.to_parquet("data/processed/train.parquet", index=False)
test_df.to_parquet("data/processed/test.parquet", index=False)

print(f"Done! Train: {len(train_df)} rows | Test: {len(test_df)} rows")