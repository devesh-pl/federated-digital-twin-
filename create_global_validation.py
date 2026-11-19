import pandas as pd
import os

print("Starting global validation dataset creation...")

# -------------------------
# LOAD HOSPITAL DATASETS
# -------------------------
h0 = pd.read_csv("datasets/hospital_0.csv")
h1 = pd.read_csv("datasets/hospital_1.csv")

print("Hospital 0 rows:", len(h0))
print("Hospital 1 rows:", len(h1))

# -------------------------
# SAMPLE SMALL VALIDATION SET
# -------------------------
h0_sample = h0.sample(frac=0.2, random_state=42)
h1_sample = h1.sample(frac=0.2, random_state=42)

# -------------------------
# COMBINE
# -------------------------
df = pd.concat([h0_sample, h1_sample])

# -------------------------
# CREATE SERVER FOLDER
# -------------------------
os.makedirs("server", exist_ok=True)

# -------------------------
# SAVE VALIDATION DATASET
# -------------------------
output_path = "server/global_validation.csv"

df.to_csv(output_path, index=False)

print(f"✅ Saved validation dataset to: {output_path}")
print("Total rows:", len(df))