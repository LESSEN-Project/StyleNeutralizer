import os
import json
import sys

from models import LLM
from prompts import get_llm_gt
from utils import parse_dataset, get_args

out_path = "preds"
os.makedirs(out_path, exist_ok=True)

MAX_NEW_TOKENS = 96
TEMPERATURE = 0.01

LLMs = ["MINISTRAL-8B-INSTRUCT", "LLAMA-3.1-8B", "GEMMA-2-9B"]
args = get_args()
dataset = parse_dataset(args.dataset)

_, retr_texts, _ = dataset.get_retr_data() 
print(f"Number of users: {len(retr_texts)}")

for model_name in LLMs:

    print(model_name)
    llm_file_name = os.path.join(out_path, f"{args.dataset}_{model_name}_zeroshot.json")

    if os.path.exists(llm_file_name):
        print("Ground truth already exists for the model!")
        continue

    llm = LLM(model_name=model_name)
    llm_gens = dict()
    for i, user in enumerate(retr_texts):

        products = retr_texts[i]
        _, ratings = dataset.get_ratings(i)

        for p, r in zip(products, ratings):

            if not llm_gens.get(p):
                llm_gens[p] = dict()

            if not llm_gens[p].get(r):   
                response = llm.prompt_chatbot(get_llm_gt(r, p, MAX_NEW_TOKENS), gen_params={"max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE})
                llm_gens[p][r] = response

        if i+1%500 == 0:
            print(f"Step: {i}")  
        sys.stdout.flush()

    print("Finished experiment!")
        
    with open(llm_file_name, "w") as f:
        json.dump(llm_gens, f)