import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Iterable, List, Optional

import spacy
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi


# ----------------------------
# Heuristic: NP (and|or) NP using POS tags
# ----------------------------
def contains_np_np_binomial(span) -> bool:
    toks = list(span)
    for i, tok in enumerate(toks):
        if tok.lower_ not in {"and", "or"}:
            continue

        left = [t for t in toks[max(0, i - 5) : i] if t.pos_ in {"NOUN", "PROPN"}]
        right = [t for t in toks[i + 1 : i + 6] if t.pos_ in {"NOUN", "PROPN"}]

        if not left or not right:
            continue

        # Exclude pronoun coordinations like "he and she"
        if any(t.pos_ == "PRON" for t in left + right):
            continue

        return True
    return False


def chunk_text(s: str, max_chars: int = 20000) -> Iterable[str]:
    """Chunk long docs so spaCy doesn't get bogged down."""
    if len(s) <= max_chars:
        yield s
        return
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        yield s[start:end]
        start = end


def save_ckpt(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_ckpt(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def list_parquet_files(
    repo_id: str,
    revision: str,
    data_dir: Optional[str],
) -> List[str]:
    """
    Return repo-relative parquet paths at the given revision.
    Optionally filter to a subdirectory (e.g., 'hacker_news', 'default', etc.)
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)

    pq = [f for f in files if f.endswith(".parquet")]
    if data_dir:
        prefix = data_dir.rstrip("/") + "/"
        pq = [f for f in pq if f.startswith(prefix)]

    if not pq:
        hint = f" (data_dir={data_dir})" if data_dir else ""
        raise RuntimeError(f"No parquet files found in {repo_id}@{revision}{hint}")

    return pq


def hf_parquet_url(repo_id: str, revision: str, path_in_repo: str) -> str:
    # Standard resolve URL works fine with load_dataset("parquet", ...)
    # Note: revision may contain slashes; requests will handle it.
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path_in_repo}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="EleutherAI/pile")
    ap.add_argument("--revision", type=str, default="refs/convert/parquet")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional subdir on the parquet branch (e.g., 'hacker_news', 'default'). "
             "Leave unset to sample across everything available in refs/convert/parquet.",
    )

    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_sent_len", type=int, default=300)
    ap.add_argument("--doc_batch", type=int, default=8)
    ap.add_argument("--max_chars_per_doc", type=int, default=20000)

    ap.add_argument("--checkpoint_every_docs", type=int, default=2000)
    ap.add_argument("--early_factor", type=int, default=30)  # stop after eligible_seen >= factor*n

    ap.add_argument("--out", type=str, default="output/pile_non_binomial_sentences.txt")
    ap.add_argument("--ckpt", type=str, default="output/pile_non_binomial_checkpoint.pkl")
    args = ap.parse_args()

    out_path = Path(args.out)
    ckpt_path = Path(args.ckpt)

    # ----------------------------
    # spaCy (POS tagging + sentence splitting; keep tagger ON)
    # ----------------------------
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    # ----------------------------
    # Resume / init
    # ----------------------------
    if ckpt_path.exists():
        state = load_ckpt(ckpt_path)
        samples = state["samples"]
        eligible_seen = state["eligible_seen"]
        file_idx = state["file_idx"]
        row_idx_in_file = state["row_idx_in_file"]
        docs_processed = state["docs_processed"]
        random.setstate(state["rng_state"])
        print(
            f"🔁 Resuming: samples={len(samples)} eligible_seen={eligible_seen:,} "
            f"file_idx={file_idx} row_idx_in_file={row_idx_in_file}"
        )
    else:
        random.seed(args.seed)
        samples = []
        eligible_seen = 0
        file_idx = 0
        row_idx_in_file = 0
        docs_processed = 0
        print("🆕 Starting fresh")

    # ----------------------------
    # Discover parquet shards (no dataset script involved)
    # ----------------------------
    print(f"📡 Listing parquet files in {args.repo}@{args.revision} ...")
    parquet_paths = list_parquet_files(args.repo, args.revision, args.data_dir)

    # Shuffle for more “random-ish” coverage early (seeded via RNG state)
    random.shuffle(parquet_paths)

    # ----------------------------
    # Iterate file-by-file so resume can skip correctly
    # ----------------------------
    pbar_files = tqdm(total=len(parquet_paths), initial=file_idx, desc="Parquet files", unit="file")

    while file_idx < len(parquet_paths):
        path_in_repo = parquet_paths[file_idx]
        url = hf_parquet_url(args.repo, args.revision, path_in_repo)

        # Stream this ONE parquet file
        ds = load_dataset(
            "parquet",
            data_files={"train": [url]},
            split="train",
            streaming=True,
        )

        # Skip rows if resuming mid-file
        it = iter(ds)
        for _ in range(row_idx_in_file):
            try:
                next(it)
            except StopIteration:
                break

        # Process rows in batches for spaCy
        buffer = []
        local_row_idx = row_idx_in_file

        for ex in it:
            text = ex.get("text")
            if not isinstance(text, str) or not text.strip():
                local_row_idx += 1
                continue

            buffer.append(text)
            local_row_idx += 1

            if len(buffer) < args.doc_batch:
                continue

            # spaCy batch
            for doc in nlp.pipe(buffer, batch_size=args.doc_batch):
                docs_processed += 1

                for piece in chunk_text(doc.text, max_chars=args.max_chars_per_doc):
                    for sent in nlp(piece).sents:
                        s = sent.text.strip()
                        if not s:
                            continue
                        if len(s) > args.max_sent_len:
                            continue
                        if contains_np_np_binomial(sent):
                            continue

                        # Eligible sentence; reservoir sample over eligible only
                        eligible_seen += 1
                        if len(samples) < args.n:
                            samples.append(s)
                        else:
                            j = random.randint(0, eligible_seen - 1)
                            if j < args.n:
                                samples[j] = s

                        if len(samples) >= args.n and eligible_seen >= args.early_factor * args.n:
                            break
                    if len(samples) >= args.n and eligible_seen >= args.early_factor * args.n:
                        break
                if len(samples) >= args.n and eligible_seen >= args.early_factor * args.n:
                    break

                # Periodic checkpoint by docs processed (more stable than sentence counts)
                if docs_processed % args.checkpoint_every_docs == 0:
                    save_ckpt(
                        ckpt_path,
                        {
                            "samples": samples,
                            "eligible_seen": eligible_seen,
                            "file_idx": file_idx,
                            "row_idx_in_file": local_row_idx,
                            "docs_processed": docs_processed,
                            "rng_state": random.getstate(),
                        },
                    )
                    tqdm.write(
                        f"💾 ckpt: samples={len(samples)} eligible_seen={eligible_seen:,} "
                        f"file_idx={file_idx} row_idx_in_file={local_row_idx}"
                    )

            buffer = []

            if len(samples) >= args.n and eligible_seen >= args.early_factor * args.n:
                break

        # Finished this file (or early stop)
        file_idx += 1
        row_idx_in_file = 0
        pbar_files.update(1)

        # Save a checkpoint at file boundary too
        save_ckpt(
            ckpt_path,
            {
                "samples": samples,
                "eligible_seen": eligible_seen,
                "file_idx": file_idx,
                "row_idx_in_file": row_idx_in_file,
                "docs_processed": docs_processed,
                "rng_state": random.getstate(),
            },
        )

        if len(samples) >= args.n and eligible_seen >= args.early_factor * args.n:
            break

    pbar_files.close()

    # ----------------------------
    # Write output
    # ----------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        for s in samples[: args.n]:
            f.write(s.replace("\n", " ") + "\n")

    print(f"✅ Wrote {min(len(samples), args.n)} sentences → {out_path}")

    # Optional: keep checkpoint; comment this out if you prefer to keep it
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("🧹 Removed checkpoint file")


if __name__ == "__main__":
    main()
