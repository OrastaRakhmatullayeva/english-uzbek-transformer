from pathlib import Path
import pandas as pd
from inference import translate

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(
    ROOT / "data/english_uzbek_100.csv"
)

df.columns = ["english", "uzbek"]
df = df.dropna().drop_duplicates()

for column in ["english", "uzbek"]:
    df[column] = df[column].astype(str).str.strip().str.lower()

df["prediction"] = df["english"].apply(translate)

df["correct"] = df["prediction"] == df["uzbek"]

accuracy = df["correct"].mean() * 100

print(f"Total samples: {len(df)}")
print(f"Correct predictions: {df['correct'].sum()}")
print(f"Training-set accuracy: {accuracy:.2f}%")

print(df.head(10).to_string(index=False))

df.to_csv(ROOT / "evaluation_results.csv", index=False)
