# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
from torch import nn
from collections import defaultdict
import pandas as pd
import re
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from torch import cuda, bfloat16
from hf_olmo import OLMoForCausalLM
#df = pd.read_csv('../../Data/nonce_binoms.csv')
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#df = df[~df['Too weird?'].isin(['maybe', 'too weird', 'yes'])]

#df = df.dropna(subset =['Sentence'])
# Create new columns 'AandB' and 'BandA'
#prompt 2: 'example: '
#prompt 3: 'instance: '
#prompt 4: 'try this: '

from torch import cuda, bfloat16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    #print("input_ids shape:", input_ids.shape)
    #print("input_ids min:", input_ids.min().item())
    #print("input_ids max:", input_ids.max().item())
    #print("Tokenizer vocab size:", tokenizer.vocab_size)
    #token_16384 = [tokenizer.decode([token_id]) for token_id in input_ids.view(-1) if token_id == 16384]
    #print("Tokens with index 16384:", token_16384)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def get_model_prefs(prompt, prompt_value, model_name, tokenizer_name):

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)
  #model.config.pad_token_id = model.config.eos_token_id
  model.eval()
  model = model.to(device)

  df = pd.read_csv('../Data/nonce_and_attested_binoms.csv')
  df['AandB'] = f"{prompt}" + df['Word1'] + ' and ' + df['Word2']
  df['BandA'] = f"{prompt}" + df['Word2'] + ' and ' + df['Word1']

  # Extracting columns for analysis
  binomial_alpha = df['AandB']
  binomial_nonalpha = df['BandA']

  #get alphabetical orderings
  input_texts_alpha = binomial_alpha

  n_batches = 40

  input_texts_alpha = np.array_split(input_texts_alpha, n_batches)
  input_texts_alpha = [x.tolist() for x in [*input_texts_alpha]]

  batch_alpha = [[]]
  timer = 0
  for minibatch in input_texts_alpha:
    timer += 1
    print(timer)
    print(minibatch)
    batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
    batch_alpha.extend(batch_placeholder)
    

  batch_alpha = batch_alpha[1:]
  sentence_probs_alpha = [sum(item[1] for item in inner_list[2:]) for inner_list in batch_alpha]

  #get nonalphabetical orderings

  input_texts_nonalpha = binomial_nonalpha

  n_batches = 40
  #input_texts_alpha = (np.array(np.array_split(input_texts_alpha, n_batches))).tolist() 

  input_texts_nonalpha = np.array_split(input_texts_nonalpha, n_batches)
  input_texts_nonalpha = [x.tolist() for x in [*input_texts_nonalpha]]

  batch_nonalpha = [[]]
  timer = 0
  for minibatch in input_texts_nonalpha:
    timer += 1
    print(timer)
    batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
    batch_nonalpha.extend(batch_placeholder)
    

  batch_nonalpha = batch_nonalpha[1:]

  sentence_probs_nonalpha = [sum(item[1] for item in inner_list[2:]) for inner_list in batch_nonalpha]

  #combine them into one dataframe

  binom_probs = {}

  for i,row in enumerate(df.itertuples()):
    binom = row[1] + ' and ' + row[2]
    binom_probs[binom] = [sentence_probs_alpha[i], sentence_probs_nonalpha[i]]


  binom_probs_df = pd.DataFrame.from_dict(binom_probs, orient = 'index', columns = ['Alpha Probs', 'Nonalpha Probs'])
  binom_probs_df.reset_index(inplace=True)
  binom_probs_df.rename(columns = {'index': 'binom'}, inplace = True)

  #write into csv


  binom_probs_df.to_csv(f'../Data/{prompt_value}_binom_ordering_prefs.csv', index = False)

#prompt 2: 'example: '
#prompt 3: 'instance: '
#prompt 4: 'try this: '
list_of_prompts = ['This next phrase must be formed compositionally: ', 'Next item: ', 'example: ', 'instance: ', 'try this: ', ''] #the last prompt works because the input is padded to begin with a begining of sentence token (token index 0 in this case)

# for prompt_value, prompt in enumerate(list_of_prompts):
#   print(prompt_value)
#   get_model_prefs(prompt=prompt, prompt_value = prompt_value)

def main():
  prompt = ''
  #"kanishka/smolm-autoreg-bpe-babylm-1e-3"
  model_names = ['kanishka/smolm-autoreg-bpe-babylm-1e-3', 'kanishka/smolm-autoreg-bpe-babylm-seed_211-1e-3', 'kanishka/smolm-autoreg-bpe-babylm-seed_1024-1e-3']
  for model_name in model_names:
    for prompt_value, prompt in enumerate(list_of_prompts):
      get_model_prefs(prompt=prompt, prompt_value = f"{prompt_value}-{model_name.split('/')[1]}", model_name = model_name, tokenizer_name = model_name) #prompt value is misleadingly named rn, it really is just how you want the .csv to be saved

main()
