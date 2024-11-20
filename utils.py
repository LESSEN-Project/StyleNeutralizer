import argparse
import os
import json
import random
import re
import numpy as np

from exp_datasets import LampDataset, AmazonDataset

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset", default="lamp_5_dev_user", type=str)
    parser.add_argument("-k", "--top_k", default=-1, type=int)
    parser.add_argument('-f', '--features', nargs='+', type=str, default=None)
    parser.add_argument("-r", "--retriever", default="contriever", type=str)
    parser.add_argument("-ce", "--counter_examples", default=None, type=int)
    parser.add_argument("-rs", "--repetition_step", default=1, type=int)

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