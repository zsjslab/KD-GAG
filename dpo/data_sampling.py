#!/usr/bin/env vLLM_ENV
import os
import sys

# 获取当前文件的绝对路径，然后找到其父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 parent_dir 添加到 Python 的模块搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os

# 添加虚拟环境的 site-packages 路径
# venv_path = "~/vllm-env/.venv"
# site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")


import argparse
import re
from rouge import Rouge
import json
import random
import itertools
from tqdm import tqdm
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import networkx as nx
from utils import eval
import jsonlines
import os
from vllm import LLM, SamplingParams
from agents.retrieval_agent import Retrieval_Agent
from agents.qa_agent import QA_Agent
from openai import OpenAI
import torch
from torch.utils.data import Dataset, DataLoader

# class ModelManager:
#     _instances = {}
    
#     @classmethod
#     def get_model(cls, model_path, device):
#         if model_path not in cls._instances:
#             cls._instances[model_path] = LLM(model=model_path)
#         return cls._instances[model_path]
    
#     @classmethod
#     def get_tokenizer(cls, model_path):
#         if model_path + "_tokenizer" not in cls._instances:
#             cls._instances[model_path + "_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
#         return cls._instances[model_path + "_tokenizer"]
    
#     @classmethod
#     def get_encoder(cls, encoder_path):
#         if encoder_path not in cls._instances:
#             cls._instances[encoder_path] = SentenceTransformer(encoder_path)
#         return cls._instances[encoder_path]

class DPODataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item.get("question", ""),
            "triples": item.get("triples", ""),
            "answer": item.get("answer", "")
        }
def collate_fn(batch):
    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HotpotQA')
    parser.add_argument('--data_file', type=str, default='teacher_kgc_res_2_.json')
    parser.add_argument('--model_path', type=str, default='../Qwen2.5-7B-Instruct')
    parser.add_argument('--encoder_path', type=str, default='../bge-large-en-v1.5')
    parser.add_argument('--use_vllm', type=bool, default=False)
    parser.add_argument('--save_file', type=str, default='sampling_data_2.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    dpo_raw= []
    with open(f"datasets/{args.dataset}/KD/{args.data_file}") as f:
        for line in f.readlines(): 
            dpo_raw.append(json.loads(line))

    if os.path.exists(f"datasets/{args.dataset}/DPO/{args.save_file}"):
        with open(f"datasets/{args.dataset}/DPO/{args.save_file}") as f:
            start = len(f.readlines())
    else:
        start = 0
        
    if args.use_vllm:
        model = LLM(model=args.model_path, max_num_seqs=4)
        # model = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
        # tokenizer = None
        # model = ModelManager.get_model(args.model_path, args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, torch_dtype=torch.float16)
    
    # tokenizer = ModelManager.get_tokenizer(args.model_path)
    # encoder = ModelManager.get_encoder(args.encoder_path) 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path) 
    encoder = SentenceTransformer(args.encoder_path)
    retriever_agent = Retrieval_Agent(encoder, model, None, tokenizer, args)

    # qa_agent = QA_Agent(model, tokenizer, args)
    
    temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    with open(f"datasets/{args.dataset}/DPO/{args.save_file}", 'a') as f:
        writer = jsonlines.Writer(f)
        for data in tqdm(dpo_raw[start: ], total=len(dpo_raw[start: ])):
            # try:
            if "question" in data:
                question = data["question"]
                triples = data["triples"]
                answer = data["answer"] 
                pruned_list = []
                res = {}
                for temperature in temperatures:
                    pruned_prompt, pruned_subgraph_res = retriever_agent.graphRetrieve(triples, question, temperature)
                    for pruned_subgraph in pruned_subgraph_res:
                        if pruned_subgraph not in pruned_list:
                            pruned_list.append(pruned_subgraph)

                    turn = 0
                    while len(pruned_list) < 3 and turn < 10:
                        pruned_prompt, pruned_subgraph_res = retriever_agent.graphRetrieve(triples, question, temperature)
                        for pruned_subgraph in pruned_subgraph_res:
                            if pruned_subgraph not in pruned_list:
                                pruned_list.append(pruned_subgraph)
                        turn += 1

                for pruned_subgraph in pruned_list:
                    response = retriever_agent.grag_answering(question, pruned_subgraph)

                    if response.lower() != 'unknown':
                        f1 = eval.f1_score(response, answer)
                        res[f1] = pruned_subgraph
                    else:
                        res[0] = pruned_subgraph
                # 选择得分最高、最低的分别作为正例、负例
                max_, max_score = res[max(list(res.keys()))], max(list(res.keys()))
                min_, min_score = res[min(list(res.keys()))], min(list(res.keys()))


                if max_score > min_score:
                    # chosen = [
                    #     {"role": "user", "content": pruned_prompt},
                    #     {"role": "system", "content": max_}
                    # ]
                    # rejected = [
                    #     {"role": "user", "content": pruned_prompt},
                    #     {"role": "system", "content": min_}
                    # ]
                    chosen = max_
                    rejected = min_
                    new = {"prompt": pruned_prompt, "chosen": chosen, "rejected": rejected, "max_score": max_score, "min_score": min_score}
                    writer.write(new)

                    continue
                
            writer.write({"wrong": "Cannot find proper positive and negative samples!"})

    # if os.path.exists(f"datasets/{args.dataset}/DPO/_add_{args.save_file}"):
    #     with open(f"datasets/{args.dataset}/DPO/_add_{args.save_file}") as f:
    #         start = len(f.readlines())
    # else:
    #     start = 0

    # old_res = []      
    # with open(f"datasets/{args.dataset}/DPO/add_{args.save_file}") as f:
    #     for line in f.readlines():
    #         old_res.append(json.loads(line))

    # with open(f"datasets/{args.dataset}/DPO/_add_{args.save_file}", 'a') as f:
    #     writer = jsonlines.Writer(f)
    #     for idx, data in tqdm(enumerate(dpo_raw[start: ]), total=len(dpo_raw[start: ])):
    #         # try:
    #         if "wrong" in old_res[start+idx].keys() or (old_res[start+idx]["max_score"] - old_res[start+idx]["min_score"] < 0.33): 
    #             if "question" in data:
    #                 question = data["question"]
    #                 triples = data["triples"]
    #                 answer = data["answer"] 
    #                 pruned_list = []
    #                 res = {}
    #                 for temperature in temperatures:
    #                     pruned_prompt, pruned_subgraph_res = retriever_agent.graphRetrieve(triples, question, temperature)
    #                     for pruned_subgraph in pruned_subgraph_res:
    #                         if pruned_subgraph not in pruned_list:
    #                             pruned_list.append(pruned_subgraph)

    #                 turn = 0
    #                 while len(pruned_list) < 4 and turn < 10:
    #                     pruned_prompt, pruned_subgraph_res = retriever_agent.graphRetrieve(triples, question, temperature)
    #                     for pruned_subgraph in pruned_subgraph_res:
    #                         if pruned_subgraph not in pruned_list:
    #                             pruned_list.append(pruned_subgraph)
    #                     turn += 1

    #                 for pruned_subgraph in pruned_list:
    #                     response = retriever_agent.grag_answering(question, pruned_subgraph)

    #                     if response.lower() != 'unknown':
    #                         f1 = eval.f1_score(response, answer)
    #                         res[f1] = pruned_subgraph
    #                     else:
    #                         res[0] = pruned_subgraph
    #                 # 选择得分最高、最低的分别作为正例、负例
    #                 max_, max_score = res[max(list(res.keys()))], max(list(res.keys()))
    #                 min_, min_score = res[min(list(res.keys()))], min(list(res.keys()))


    #                 if max_score > min_score:
    #                     chosen = max_
    #                     rejected = min_
    #                     new = {"prompt": pruned_prompt, "chosen": chosen, "rejected": rejected, "max_score": max_score, "min_score": min_score}
    #                     writer.write(new)

    #                     continue
                    
    #             writer.write({"wrong": "Cannot find proper positive and negative samples!"})    
    #         else:
    #             writer.write(old_res[start+idx])    
    

                

                    
            