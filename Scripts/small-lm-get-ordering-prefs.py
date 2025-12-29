# ==========================================================
#  IMPORTS
# ==========================================================
import os
import torch
import pandas as pd
import numpy as np
from torch import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 
import glob
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


@torch.no_grad()
def to_tokens_and_logprobs(model, tokenizer, input_texts):

    enc = tokenizer(input_texts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1)

    logprobs = logprobs[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * target_mask

    return token_logprobs.sum(dim=-1).tolist()


def get_model_prefs(prompt, model_name, tokenizer, model):

    df = pd.read_csv('../Data/nonce_and_attested_binoms.csv')

    df['AandB'] = prompt + df['Word1'] + ' and ' + df['Word2']
    df['BandA'] = prompt + df['Word2'] + ' and ' + df['Word1']

    binomial_alpha = df['AandB'].tolist()
    binomial_nonalpha = df['BandA'].tolist()

    seqs_alpha = np.array_split(binomial_alpha, 5)
    alpha_scores = []
    for batch in seqs_alpha:
        alpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    seqs_nonalpha = np.array_split(binomial_nonalpha, 5)
    nonalpha_scores = []
    for batch in seqs_nonalpha:
        nonalpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    rows = []
    for i, row in enumerate(df.itertuples()):
        rows.append({
            "model": model_name,
            "prompt": prompt,
            "binom": f"{row.Word1} and {row.Word2}",
            "alpha_logprob": alpha_scores[i],
            "nonalpha_logprob": nonalpha_scores[i],
            "preference": alpha_scores[i] - nonalpha_scores[i]
        })

    return pd.DataFrame(rows)



# ==========================================================
#  MAIN LOOP
# ==========================================================
list_of_prompts = [
    "Well, the ",
    "So, the ",
    "Then the ",
    "Possibly the ",
    "Or even the ",
    "Maybe the ",
    "Perhaps the ",
    "At times the ",
    "Suddenly, the ",
    "Honestly just the "
    # "Especially the ",
    # "For instance ",
    # "In some cases ",
    # "Every now and then ",
    # "Occasionally you’ll find ",
    # "There can be examples like ",
    # "You might notice things like ",
    # "People sometimes mention ",
    # "Sometimes just ",
    # "Nothing specific comes to mind except the ",
    # "It reminded me loosely of the ",
    # "There was a vague reference to the ",
    # "Unexpectedly the ",
    # "It’s easy to overlook the ",
    # "There used to be talk of ",
    # "Out in the distance was the ",
    # "What puzzled everyone was the ",
    # "At some point I overheard ",
    # "Without warning came ",
    # "A friend once described the ",
    # "The scene shifted toward ",
    # "Nobody expected to hear about ",
    # "Things eventually turned toward ",
    # "The conversation eventually returned to ",
    # "I only remember a hint of the ",
    # "I couldn’t quite place the ",
    # "It somehow led back to the ",
    # "What stood out most was the ",
    # "The oddest part involved the ",
    # "Later on, people were discussing ",
    # "There was this fleeting idea about ",
    # "I once heard someone bring up the ",
    # "There was a moment involving the ",
    # "It all started when we noticed the ",
    # "Another example floated around concerning the ",
    # "I came across something about the ",
    # "A situation arose involving the ",
    # "The conversation drifted toward the ",
    # "At one point we ended up discussing ",
    # "Out of nowhere came a mention of the "
]

def main():

    epochs = ["1", "5", "10"]
    modelsizes = ["70M", "160M", "410M"]
    conditions = ["genpref", "handcoded", "relfreq"]
    binomsizes = ["n1000", "n5000", "n10000", "nunique"]
    nonbinoms = ["0", "50k", "150k", "300k"]

    model_names = [
        f"qing-yao/{condition}_{binomsize}_nb{nonbinom}_{modelsize}_ep{epoch}_lr1e-4_seed42"
        for condition in conditions
        for binomsize in binomsizes
        for nonbinom in nonbinoms
        for modelsize in modelsizes
        for epoch in epochs
    ]

    model_names_base = [
        f"qing-yao/baseline_nb{nonbinom}_{modelsize}_ep{epoch}_lr1e-4_seed42"
        for nonbinom in ["50k", "150k", "300k"]
        for modelsize in modelsizes
        for epoch in epochs
    ]

    
    model_names.extend([
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m"
    ])

    model_names.extend(model_names_base)
    print(len(model_names))

    out_dir = "../Data/model_results"
    os.makedirs(out_dir, exist_ok=True)
    failed_models = []


    # --------------------------------------------------
    # 🔍 Find already–completed models
    # --------------------------------------------------
    existing_files = glob.glob(os.path.join(out_dir, "*.csv"))
    done_models = sorted(os.path.splitext(os.path.basename(f))[0] for f in existing_files)

    print("\n==============================================")
    print(f"📦 Found {len(done_models)} completed model outputs:")
    for m in done_models:
        print(f"   ✔ {m}")
    print("==============================================\n")

    for model_name in model_names:
        safe_id = model_name.split("/")[-1]
        out_path = os.path.join(out_dir, f"{safe_id}.csv")

        # Skip existing
        if os.path.exists(out_path):
            print(f"⏭  Skipping {safe_id} — results already exist")
            continue

        print(f"\n🚀 Running {safe_id}")

        try:
            # -------- LOAD TOKENIZER --------
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # -------- LOAD MODEL --------
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32
            ).to(device).eval()

        except Exception as e:
            msg = f"FAILED TO LOAD: {safe_id} — {e}"
            print(f"\n🚨 {msg}\n➡️  Continuing…\n")
            failed_models.append(msg)
            continue  # <-- go to next model!

        try:
            # -------- RUN SCORING --------
            per_prompt_results = []
            for prompt in tqdm(list_of_prompts, desc=f"{safe_id} prompts"):
                df_out = get_model_prefs(prompt, model_name, tokenizer, model)
                per_prompt_results.append(df_out)

            final_df = pd.concat(per_prompt_results, ignore_index=True)

            # atomic write protection
            tmp_path = out_path + ".tmp"
            final_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)

            print(f"💾 Saved {safe_id} → {out_path}")

        except Exception as e:
            msg = f"FAILED DURING EVAL: {safe_id} — {e}"
            print(f"\n🚨 {msg}\n➡️  Continuing…\n")
            failed_models.append(msg)

        finally:
            # cleanup GPU memory
            try:
                del model
                torch.cuda.empty_cache()
            except:
                pass

    print("\n==============================================")
    print("🏁 RUN COMPLETE")
    print("==============================================")

    if failed_models:
        print(f"\n⚠️ {len(failed_models)} models failed:\n")
        for m in failed_models:
            print("  •", m)
    else:
        print("\n🎉 No model failures — you’re all good!\n")

    log_path = "../Data/model_results/failed_models.log"

    if failed_models:
        with open(log_path, "w", encoding="utf-8") as f:
            for m in failed_models:
                f.write(m + "\n")

        print(f"\n📝 Wrote failure log → {log_path}")
    else:
        print("\n🎉 No model failures — you’re all good!\n")





if __name__ == "__main__":
    main()
    files = glob.glob("../Data/model_results/*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv("../Data/grid_search_results.csv", index=False)
