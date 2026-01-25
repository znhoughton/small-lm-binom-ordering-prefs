from datasets import load_dataset
from collections import Counter
import re
from tqdm import tqdm
import pandas as pd

# Load the English BabyLM dataset
ds = load_dataset("qing-yao/slightly-cleaner-babylm", split="train", streaming=True)

token_re = re.compile(r"\b\w+\b")
counter = Counter()

for ex in tqdm(ds, desc="Counting BabyLM unigrams"):
    text = ex.get("text", "")
    tokens = token_re.findall(text.lower())
    counter.update(tokens)

# Convert to DataFrame
unigrams = (
    pd.DataFrame(counter.items(), columns=["word", "count"])
      .sort_values("count", ascending=False)
)

unigrams.to_csv("../Data/babylm_eng_unigrams.csv", index=False)
print("Saved babylm_eng_unigrams.csv")
