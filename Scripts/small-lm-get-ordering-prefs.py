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

    outputs = model(input_ids)
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
    'This next phrase must be formed compositionally: ',
    'Next item: ',
    'example: ',
    'instance: ',
    'try this: ',
    ''
]

def main():
    model_names = [
        'gpt2-xl',
        'kanishka/smolm-autoreg-bpe-babylm-1e-3',
        'kanishka/smolm-autoreg-bpe-babylm-seed_211-1e-3',
        'kanishka/smolm-autoreg-bpe-babylm-seed_1024-1e-3'
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
    out_path = '../Data/ALL_MODELS_ALL_PROMPTS_BINOMIAL_PREFERENCES.csv'
    #change to ALL_MODELS_ALL_PROMPTS.csv when just want nonce binoms
    final_df.to_csv(out_path, index=False)

    print(f"\n✔ MASTER CSV saved: {out_path}")
    print(f"✔ Total rows: {len(final_df)}")


main()
