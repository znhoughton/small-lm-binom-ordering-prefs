from datasets import load_dataset
from collections import Counter
import re
from tqdm import tqdm
import pandas as pd

ds = load_dataset(
    "qing-yao/slightly-cleaner-babylm",
    split="train",
    streaming=True
)

token_re = re.compile(r"\b\w+\b")
counter = Counter()

for ex in tqdm(ds, desc="Counting BabyLM trigrams"):
    text = ex.get("text", "")
    tokens = token_re.findall(text.lower())

    for i in range(len(tokens) - 2):
        trigram = " ".join(tokens[i:i+3])
        counter[trigram] += 1

trigrams = (
    pd.DataFrame(counter.items(), columns=["trigram", "count"])
      .sort_values("count", ascending=False)
)

trigrams.to_csv("../Data/babylm_eng_trigrams.csv", index=False)
print("Saved babylm_eng_trigrams.csv")
