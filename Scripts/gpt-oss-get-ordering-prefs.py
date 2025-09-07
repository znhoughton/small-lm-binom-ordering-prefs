import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from auto_gptq import AutoGPTQForCausalLM

logging.basicConfig(
    level=logging.DEBUG,  # or INFO if DEBUG is too verbose
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="debug_logprobs.log",
    filemode="w"
)
    
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)

    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # shift so we align probs[t] with input_ids[t+1]
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for sent_idx, (input_sentence, input_probs) in enumerate(zip(input_ids, gen_probs)):
        text_sequence = []
        logging.debug(f"Sentence {sent_idx}:")
        for tok_idx, (token, p) in enumerate(zip(input_sentence, input_probs)):
            if token not in tokenizer.all_special_ids:
                decoded = tokenizer.decode(token)
                logging.debug(
                    f"  Token {tok_idx}: {decoded!r} "
                    f"(id={token.item()}), log_prob={p.item():.4f}"
                )
                text_sequence.append((decoded, p.item()))
        # Log the sum of log-probs (skipping first 2 like your later code)
        if len(text_sequence) > 2:
            sum_val = sum(prob for _, prob in text_sequence[2:])
            logging.debug(f"  → Sum of log_probs (excluding first 2 tokens): {sum_val:.4f}")
        else:
            logging.debug("  → Not enough tokens to compute sum.")
        batch.append(text_sequence)

    return batch

def get_sentence_probs(model, tokenizer, input_texts, n_batches=40):
    sentence_probs = []
    for i, minibatch in enumerate(np.array_split(input_texts, n_batches), 1):
        print(f"Processing batch {i}/{n_batches}")
        token_probs = to_tokens_and_logprobs(model, tokenizer, minibatch.tolist())
        sentence_probs.extend([sum(item[1] for item in inner_list[2:]) for inner_list in token_probs])
    return sentence_probs
    

def run_experiment(model, tokenizer, prompt, prompt_value, input_file="data.csv", out_dir="./"):
    df = pd.read_csv(input_file)
    df["AandB"] = prompt + df["Word1"] + " and " + df["Word2"]
    df["BandA"] = prompt + df["Word2"] + " and " + df["Word1"]

    df["alpha_prob"] = get_sentence_probs(model, tokenizer, df["AandB"])
    df["nonalpha_prob"] = get_sentence_probs(model, tokenizer, df["BandA"])

    out_path = os.path.join(out_dir, f"{prompt_value}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


def main():
    model_names = [
        #"openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
    prompts = [
        "This next phrase must be formed compositionally: ",
        "Next item: ",
        "example: "
        #"instance: ",
        #"try this: ",
        #""
    ]
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        for i, prompt in enumerate(prompts):
            run_experiment(model, tokenizer, prompt, f"{i}-{model_name.split('/')[1]}")
     

if __name__ == "__main__":
    main()