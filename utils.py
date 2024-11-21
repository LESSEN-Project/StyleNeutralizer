import argparse
import os
import json
import random
import re
import numpy as np

from exp_datasets import LampDataset, AmazonDataset

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset", default="amazon_Grocery_and_Gourmet_Food_2018", type=str)
    parser.add_argument("-ce", "--counter_examples", default=None, type=int)
    parser.add_argument("-rs", "--repetition_step", default=1, type=int)

    return parser.parse_args()

def get_bfi_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset", default="amazon_Grocery_and_Gourmet_Food_2018", type=str)

    return parser.parse_args()

def parse_dataset(dataset):

    if dataset.startswith("lamp"):
        num = int(dataset.split("_")[1])
        data_split = dataset.split("_")[2]
        split = dataset.split("_")[-1]
        return LampDataset(num, data_split, split)
    elif dataset.startswith("amazon"):
        year = int(dataset.split("_")[-1])
        category = "_".join(dataset.split("_")[1:-1])
        return AmazonDataset(category, year)
    else:
        raise Exception("Dataset not known!")
    
def oai_get_or_create_file(client, filename):

    files = client.files.list()
    existing_file = next((file for file in files if file.filename == filename), None)

    if existing_file:
        print(f"File '{filename}' already exists. File ID: {existing_file.id}")
        return existing_file.id
    else:
        with open(os.path.join("preds", filename), "rb") as file_data:
            new_file = client.files.create(
                file=file_data,
                purpose="batch"
            )
        print(f"File '{filename}' created. File ID: {new_file.id}")
        return new_file.id
    
def oai_get_batch_res(client):

    batches = client.batches.list()
    files = client.files.list()

    for batch in batches:
        
        filename = [file.filename for file in files if file.id == batch.input_file_id]

        if filename:

            merged_res = []
            path_to_file = os.path.join("preds", filename[0])

            data = []
            with open(path_to_file, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            batch_res = client.files.content(batch.output_file_id).text
            batch_res = [json.loads(line) for line in batch_res.splitlines()]

            for sample in data:
                res = [res["response"]["body"]["choices"][0]["message"]["content"] for res in batch_res if res["custom_id"] == sample["custom_id"]]
                merged_res.append({
                    "id": sample["custom_id"],
                    "output": res[0].strip(),
                    "prompt": sample["body"]["messages"][0]["content"],
                    "model_inf_time": "n/a", 
            })
            with open(os.path.join("preds", f"{filename[0].split('.')[0]}.json"), "w") as f:
                task = f"LaMP_{filename[0].split('_')[1]}" if filename[0].startswith("lamp") else "_".join(filename[0].split("_")[:-3])
                json.dump({
                    "task": task,
                    "golds": merged_res
                }, f)