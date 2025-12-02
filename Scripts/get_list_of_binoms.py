

### create binomials corpus


import requests
import gzip
import re
from collections import Counter, defaultdict
from typing import Dict, Tuple, Generator, Optional

import nltk
from nltk.corpus import cmudict, wordnet as wn
import pandas as pd
CHECKPOINT_FILE = "binomials_checkpoint.json"
SAVE_EVERY = 25000000000          # lines
SAVE_BINOMIALS_EVERY = 100000  # binomials
# ==========================================================
#  0. NLTK SETUP (RUN ONCE)
# ==========================================================
# You can comment these out after the resources are downloaded.
nltk.download("cmudict")
nltk.download("wordnet")
nltk.download("omw-1.4")

CMU = cmudict.dict()


# ==========================================================
#  1. HELPERS: WORD FILTERS
# ==========================================================
def is_alpha_word(w: str) -> bool:
    """True if token is all lowercase a–z."""
    return bool(re.fullmatch(r"[a-z]+", w))


_noun_cache: Dict[str, bool] = {}


def is_noun_wordnet(w: str) -> bool:
    """
    Heuristic noun check using WordNet.
    Cached for efficiency.
    """
    if w in _noun_cache:
        return _noun_cache[w]
    syns = wn.synsets(w)
    is_noun = any(s.pos() == "n" for s in syns)
    _noun_cache[w] = is_noun
    return is_noun


def has_cmu_pron(w: str) -> bool:
    return w in CMU


# ==========================================================
#  2. GOOGLE NGRAM 3-GRAM STREAMING
# ==========================================================
BASE = "http://storage.googleapis.com/books/ngrams/books/"
PREFIX_3GRAM = "googlebooks-eng-all-3gram-20120701-"

import requests
import gzip
from collections import defaultdict
import requests, gzip, re

UNIGRAM_BASE = "http://storage.googleapis.com/books/ngrams/books/"
unigram_cache = {}             # per-word counts
prefix_cache = {}              # NEW: per-letter counts {prefix → dict{word → count}}
from tqdm import tqdm


def get_bulk_unigram_counts(words):
    """
    Efficiently compute unigram match counts for a *set* of words.
    Groups words by first letter, scans each 1-gram file once.
    Stores results in global unigram_cache.

    words : iterable of strings (lowercase)
    """
    global unigram_cache
    # Filter out words we already cached
    words = {w.lower() for w in words}
    words_needed = [w for w in words if w not in unigram_cache]
    if not words_needed:
        return unigram_cache
    # Group by prefix (first letter)
    groups = {}
    for w in words_needed:
        prefix = w[0]
        groups.setdefault(prefix, []).append(w)
    print(f"📚 Need unigram counts for {len(words_needed)} words")
    print(f"🔤 Prefix groups to process: {sorted(groups.keys())}")
    for prefix, words_for_prefix in groups.items():
        fname = f"googlebooks-eng-all-1gram-20120701-{prefix}.gz"
        url = UNIGRAM_BASE + fname
        print(f"\n📂 Fetching 1-gram file for prefix '{prefix}': {fname}")
        print(f"   Words to find in this file: {len(words_for_prefix)}")
        # Prepare a fast lookup set for cleaned forms
        target_set = set(words_for_prefix)
        # Initialize counts
        for w in words_for_prefix:
            unigram_cache[w] = 0
        try:
            with requests.get(url, stream=True, timeout=None) as r:
                r.raise_for_status()
                f = gzip.GzipFile(fileobj=r.raw)
                # Line-by-line scan with progress bar
                for rawline in tqdm(f, desc=f"   📄 Scanning {prefix}.gz"):
                    try:
                        line = rawline.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 4:
                        continue
                    ngram, year, match_count, book_count = parts
                    cleaned = re.sub(r"[^a-z]", "", ngram.lower())
                    if cleaned in target_set:
                        unigram_cache[cleaned] += int(match_count)
        except Exception as e:
            print(f"⚠️ Error retrieving file for prefix '{prefix}': {e}")
    print("\n✅ Done. Unigram counts cached for all requested words.")
    return unigram_cache



def load_prefix_counts(prefix: str):
    """Load ALL unigram counts for words starting with prefix."""
    if prefix in prefix_cache:
        return prefix_cache[prefix]
    fname = f"googlebooks-eng-all-1gram-20120701-{prefix}.gz"
    url = UNIGRAM_BASE + fname
    counts = defaultdict(int)
    print(f"📥 Loading 1-gram prefix file: {prefix}")
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        f = gzip.GzipFile(fileobj=r.raw)
        for rawline in f:
            try:
                line = rawline.decode("utf-8")
            except:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            ngram, year, match_count, volume_count = parts
            # normalize
            cleaned = re.sub(r'[^a-z]', '', ngram.lower())
            if cleaned.startswith(prefix):
                try:
                    counts[cleaned] += int(match_count)
                except:
                    pass
    prefix_cache[prefix] = counts
    return counts


def get_unigram_match_count(word: str) -> int:
    """Fast unigram lookup using prefix file caching."""
    word = word.lower()
    # If already known, return immediately
    if word in unigram_cache:
        return unigram_cache[word]
    prefix = word[0]  # one letter
    # Load the entire prefix file once
    prefix_counts = load_prefix_counts(prefix)
    # Now lookup is instant (dictionary lookup)
    count = prefix_counts.get(word, 0)
    unigram_cache[word] = count
    return count




def iter_3gram_suffixes() -> Generator[str, None, None]:
    """
    Generate file suffixes for English 3-grams.
    We focus on alphabetic prefixes, which is where normal words live.

    Example file names that this will try:
        googlebooks-eng-all-3gram-20120701-a_.gz
        googlebooks-eng-all-3gram-20120701-aa.gz
        googlebooks-eng-all-3gram-20120701-ab.gz
        ...
        googlebooks-eng-all-3gram-20120701-z_.gz
        googlebooks-eng-all-3gram-20120701-zz.gz

    Some suffixes may not exist; those will just 404 and be skipped.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    # First: single-letter + underscore buckets (a_, b_, ..., z_)
    for ch in letters:
        yield f"{ch}_"
    # Then: two-letter buckets (aa, ab, ..., zz)
    for ch1 in letters:
        for ch2 in letters:
            yield f"{ch1}{ch2}"

def stream_trigram_file(suffix: str):
    fname = f"{PREFIX_3GRAM}{suffix}.gz"
    url = BASE + fname
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except:
        return None
    def _gen():
        import zlib
        decompressor = zlib.decompressobj(16+zlib.MAX_WBITS)
        # stream chunks instead of whole gzip
        for chunk in r.raw.stream(1024*1024, decode_content=False):
            try:
                data = decompressor.decompress(chunk)
            except Exception:
                continue

            for line in data.splitlines():
                yield line.decode("utf-8", errors="ignore")
    return _gen()




def parse_3gram_line(line: str) -> Optional[Tuple[str, int]]:
    """
    Parse a Google 3-gram line.

    Format (conceptually):
        ngram<TAB>year<TAB>match_count<TAB>volume_count

    We return (ngram_string, match_count) or None on failure.
    """
    line = line.strip()
    if not line:
        return None
    # Try tab-split first (official format)
    parts = line.split("\t")
    if len(parts) == 4:
        ngram_str, year, match_count, volume_count = parts
    else:
        # Fallback: some viewers show it with spaces
        parts = re.split(r"\s+", line)
        if len(parts) < 4:
            return None
        ngram_str = " ".join(parts[:-3])
        year, match_count, volume_count = parts[-3:]
    try:
        count = int(match_count)
    except ValueError:
        return None
    return ngram_str, count


# ==========================================================
#  3. MAIN: COLLECT BINOMIALS “A and B”
# ==========================================================
from tqdm import tqdm
def collect_binomials_from_google_ngrams(
    target_pairs: int | None = None,
    min_binomial_freq: int = 5,
    require_nouns: bool = True,
) -> pd.DataFrame:

    print("🚀 Starting Google Ngram streaming & binomial extraction...\n")

    # ---- TRY TO RESUME FROM CHECKPOINT ----
    resume = load_checkpoint()

    if resume:
        (
            suffix_resume,
            total_lines_seen,
            binom_freq,
            collected,
        ) = resume

        print(f"⏯ Resuming from suffix='{suffix_resume}' "
              f"after {total_lines_seen:,} lines…")

    else:
        suffix_resume = None
        total_lines_seen = 0
        binom_freq = Counter()
        collected = {}

    last_saved_binomials = len(collected)
    total_files_attempted = 0
    word_freq: Counter[str] = Counter()

    print("\n---- MAIN LOOP ----")
    skipping = suffix_resume is not None   # <-- add this

    for suffix_num, suffix in enumerate(iter_3gram_suffixes(), start=1):

        # This replaces the lexicographic version:
        if skipping:
            if suffix == suffix_resume:
                skipping = False
            continue

        total_files_attempted += 1
        print(f"\n📂 [{suffix_num}] Opening file suffix: {suffix}")

        trigram_gen = stream_trigram_file(suffix)
        if trigram_gen is None:
            print(f"   ⚠️ File for suffix '{suffix}' does NOT exist. Skipping.")
            continue

        pbar = tqdm(
            trigram_gen,
            desc=f"   📄 Streaming {suffix}",
            unit="lines",
            mininterval=0.5,
        )

        # ---- STREAM AND PROCESS EACH LINE ----
        for line in pbar:
            total_lines_seen += 1
            parsed = parse_3gram_line(line)
            if parsed is None:
                continue

            ngram_str, count = parsed
            parts = ngram_str.split()
            if len(parts) != 3:
                continue

            w1, w2, w3 = (w.lower() for w in parts)

            if w2 != "and":
                continue

            if not (is_alpha_word(w1) and is_alpha_word(w3)):
                continue

            # optional noun filtering
            if require_nouns and not (
                is_noun_wordnet(w1) and is_noun_wordnet(w3)
            ):
                continue

            # CMU pron check
            if not (has_cmu_pron(w1) and has_cmu_pron(w3)):
                continue

            # ---- UPDATE COUNTS ----
            binom_freq[(w1, w3)] += count
            word_freq[w1] += count
            word_freq[w3] += count

            # if new and reaches minimum freq
            if (
                binom_freq[(w1, w3)] >= min_binomial_freq
                and (w1, w3) not in collected
            ):
                collected[(w1, w3)] = True

                if len(collected) % 10 == 0:
                    print(f"   ✅ {len(collected)} binomials collected "
                          f"(suffix={suffix}, lines={total_lines_seen:,})")

                # stop early if requested
                if target_pairs and len(collected) >= target_pairs:
                    pbar.close()
                    return _binomial_dict_to_df(binom_freq, word_freq, collected)

            # ---- PERIODIC CHECKPOINT SAVING ----
            if total_lines_seen % SAVE_EVERY == 0:
                save_checkpoint(suffix, total_lines_seen, binom_freq, collected)

            if len(collected) >= last_saved_binomials + SAVE_BINOMIALS_EVERY:
                save_checkpoint(suffix, total_lines_seen, binom_freq, collected)
                last_saved_binomials = len(collected)

        print(f"   ✔ Finished scanning file suffix: {suffix}")

    print("\n⚠️ Exhausted all files.")
    print(f"📊 Total lines scanned: {total_lines_seen:,}")
    print(f"📄 Total files attempted: {total_files_attempted}")

    return _binomial_dict_to_df(binom_freq, word_freq, collected)


from tqdm import tqdm
def _binomial_dict_to_df(
    binom_freq: Counter,
    word_freq: Counter,
    collected: Dict[Tuple[str, str], bool],
) -> pd.DataFrame:
    """
    Convert collected binomials into a dataframe with unigram frequencies,
    AND consolidate reversed binomials.

    Example:
        'bread and butter'  = 10
        'butter and bread'  = 5

    Becomes:
        WordA=bread, WordB=butter,
        AlphaOrderFreq=10,
        NonAlphaOrderFreq=5,
        TotalFreq=15
    """
    # ------------------------------------------
    # 1. Consolidate AB and BA forms
    # ------------------------------------------
    combined = {}  # (alphaA, alphaB) → {"alpha":count, "nonalpha":count}
    for (w1, w3), count in binom_freq.items():
        A = min(w1, w3)
        B = max(w1, w3)
        is_alpha_order = (w1 == A)
        if (A, B) not in combined:
            combined[(A, B)] = {"alpha": 0, "nonalpha": 0}
        if is_alpha_order:
            combined[(A, B)]["alpha"] += count
        else:
            combined[(A, B)]["nonalpha"] += count
    # ------------------------------------------
    # 2. Extract ALL unique words that appear
    # ------------------------------------------
    all_words = set()
    for (A, B) in combined:
        all_words.add(A)
        all_words.add(B)
    print(f"\n🔎 Fetching unigram counts for {len(all_words)} words…")
    get_bulk_unigram_counts(all_words)
    # ------------------------------------------
    # 3. Build dataframe rows
    # ------------------------------------------
    rows = []
    for (A, B), info in combined.items():
        alpha_count = info["alpha"]
        nonalpha_count = info["nonalpha"]
        total_count = alpha_count + nonalpha_count
        rows.append(
            {
                "WordA": A,
                "WordB": B,
                "AlphaOrderFreq": alpha_count,
                "NonAlphaOrderFreq": nonalpha_count,
                "TotalFreq": total_count,
                "WordA_Unigram": unigram_cache.get(A, 0),
                "WordB_Unigram": unigram_cache.get(B, 0),
                "CMU_A": " ".join(CMU[A][0]) if A in CMU else "",
                "CMU_B": " ".join(CMU[B][0]) if B in CMU else "",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("TotalFreq", ascending=False).reset_index(drop=True)
    return df


import json
import os

def save_checkpoint(
    suffix, total_lines_seen, binom_freq, collected
):
    # convert tuple keys to strings for JSON
    binom_freq_json = {f"{k[0]}|||{k[1]}": v for k,v in binom_freq.items()}

    state = {
        "suffix": suffix,
        "total_lines_seen": total_lines_seen,
        "binom_freq": binom_freq_json,
        "collected": [f"{a}|||{b}" for (a,b) in collected]
    }

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)

    print(f"\n💾 CHECKPOINT SAVED at suffix={suffix} lines={total_lines_seen}")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    with open(CHECKPOINT_FILE) as f:
        state = json.load(f)

    # reconstruct tuples
    def unstring(x):
        return tuple(x.split("|||"))

    binom_freq = Counter({unstring(k): v for k, v in state["binom_freq"].items()})
    collected = {unstring(k): True for k in state["collected"]}

    return (
        state["suffix"],
        state["total_lines_seen"],
        binom_freq,
        collected,
    )

# ==========================================================
#  4. EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    # This will take a while the first time, since it streams a lot of data.
    # You can start with smaller target_pairs just to test (e.g., 50 or 100).



    df_binomials = collect_binomials_from_google_ngrams(
        target_pairs=None,
        min_binomial_freq=100,
        require_nouns=True,
    )
    print(df_binomials.head())
    df_binomials.to_csv("../Data/google_binomials_cmudict.csv", index=False)
    print("💾 Saved to google_binomials_cmudict.csv")

