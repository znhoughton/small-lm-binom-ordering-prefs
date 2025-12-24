# ==========================================================
#  IMPORTS
# ==========================================================
import os
import torch
import pandas as pd
import numpy as np
from torch import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm   # <-- NEW

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


# ==========================================================
#  LOG-PROBABILITY FUNCTION
# ==========================================================
@torch.no_grad()
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    """
    Returns ONE summed log-prob per input sequence.
    """

    enc = tokenizer(input_texts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask = attention_mask)
    logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1)

    logprobs = logprobs[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * target_mask

    seq_scores = token_logprobs.sum(dim=-1).tolist()
    return seq_scores


# ==========================================================
#  MAIN FUNCTION FOR COMPUTING MODEL SCORES
# ==========================================================
def get_model_prefs(prompt, prompt_value, model_name, tokenizer_name):

    print(f"\n=== Running model: {model_name} | prompt '{prompt}' ===")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # GPT-2 needs a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    model.eval()
    model.to(device)

    df = pd.read_csv('../Data/nonce_and_attested_binoms.csv')
    #changed to: nonce_binoms.csv for just novel binomials
    df['AandB'] = f"{prompt}" + df['Word1'] + ' and ' + df['Word2']
    df['BandA'] = f"{prompt}" + df['Word2'] + ' and ' + df['Word1']

    binomial_alpha = df['AandB'].tolist()
    binomial_nonalpha = df['BandA'].tolist()

    # ==================================================
    #  Compute ALPHA logprobs (with progress bar)
    # ==================================================
    print("→ Computing ALPHA logprobs...")
    seqs_alpha = np.array_split(binomial_alpha, 40)
    alpha_scores = []

    for batch in tqdm(seqs_alpha, desc=f"{model_name} alpha batches"):
        alpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    # ==================================================
    #  Compute NON-ALPHA logprobs (with progress bar)
    # ==================================================
    print("→ Computing NON-ALPHA logprobs...")
    seqs_nonalpha = np.array_split(binomial_nonalpha, 40)
    nonalpha_scores = []

    for batch in tqdm(seqs_nonalpha, desc=f"{model_name} nonalpha batches"):
        nonalpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    # ==================================================
    #  Build DataFrame for this (model, prompt)
    # ==================================================
    rows = []
    for i, row in enumerate(df.itertuples()):
        binom = f"{row.Word1} and {row.Word2}"
        rows.append({
            "model": model_name,
            "prompt": prompt,
            "binom": binom,
            "alpha_logprob": alpha_scores[i],
            "nonalpha_logprob": nonalpha_scores[i],
            "preference": alpha_scores[i] - nonalpha_scores[i]
        })

    return pd.DataFrame(rows)


# ==========================================================
#  MAIN LOOP
# ==========================================================
list_of_prompts = [
    "Well, ",
    "So, ",
    "Then ",
    "Possibly ",
    "Or even ",
    "Maybe a ",
    "Perhaps a ",
    "At times ",
    "Suddenly, the ",
    "Honestly just ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you’ll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It’s easy to overlook the ",
    "There used to be talk of ",
    "Out in the distance was the ",
    "What puzzled everyone was the ",
    "At some point I overheard ",
    "Without warning came ",
    "A friend once described the ",
    "The scene shifted toward ",
    "Nobody expected to hear about ",
    "Things eventually turned toward ",
    "The conversation eventually returned to ",
    "I only remember a hint of the ",
    "I couldn’t quite place the ",
    "It somehow led back to the ",
    "What stood out most was the ",
    "The oddest part involved the ",
    "Later on, people were discussing ",
    "There was this fleeting idea about ",
    "I once heard someone bring up the ",
    "There was a moment involving the ",
    "It all started when we noticed the ",
    "Another example floated around concerning the ",
    "I came across something about the ",
    "A situation arose involving the ",
    "The conversation drifted toward the ",
    "At one point we ended up discussing ",
    "Out of nowhere came a mention of the "
]

def main():
    epochs = ["10", "18", "30"]
    lrs = ["3e-5", "5e-5", "1e-4"]

    model_names = [
        #f"qing-yao/genfreq-finetuned-ep{epoch}_seed-42_{lr}" #base model is qing-yao/binomial-babylm-base_seed-42_1e-3
        #for epoch in epochs
        #for lr in lrs
        #"qing-yao/relfreq-finetuned-ep10_seed-42_1e-4"
        #"qing-yao/genfreq-finetuned-ep1_seed-42_1e-4"
        #"qing-yao/genfreq-finetuned_seed-42_5e-5"
        #"qing-yao/handcoded-finetuned-ep1_n1000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep1_n2000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep1_n3000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep1_n4000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep1_n5000_seed-42_1e-4"
        #"qing-yao/handcoded-finetuned-ep2_n5000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep5_n5000_seed-42_1e-4",
        #"qing-yao/handcoded-finetuned-ep10_n5000_seed-42_1e-4"
        #"EleutherAI/pythia-160m",
        #"qing-yao/handcoded-finetuned-ep10_seed-42_1e-4",
        #"EleutherAI/pythia-410m",
        #"EleutherAI/pythia-1b"
        "qing-yao/genpref-finetuned-ep10_seed-42_1e-4"
    ]

    all_results = []

    for model_name in model_names:
        safe_id = model_name.split("/")[-1]

        for prompt_value, prompt in enumerate(tqdm(list_of_prompts, desc=f"Prompts for {safe_id}")):
            df_out = get_model_prefs(
                prompt=prompt,
                prompt_value=f"{prompt_value}-{safe_id}",
                model_name=model_name,
                tokenizer_name=model_name
            )
            all_results.append(df_out)

    # ================================================
    #  SAVE MASTER CSV
    # ================================================
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = '../Data/PYTHIA_SAMPLED_10EP_MODEL_BINOMIAL_PREFERENCES.csv'
    #change to ALL_MODELS_ALL_PROMPTS.csv when just want nonce binoms
    final_df.to_csv(out_path, index=False)

    print(f"\n✔ MASTER CSV saved: {out_path}")
    print(f"✔ Total rows: {len(final_df)}")


main()
