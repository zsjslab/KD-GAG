from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
import sys

# 获取当前文件的绝对路径，然后找到其父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 parent_dir 添加到 Python 的模块搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils.kgc import KGC
from utils.kgc_refine import KGCRefine
from utils.embedding_database import EmbeddingDatabase
# from utils.visualization import KgVisualize
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import argparse
# import pygraphviz as pgv
# from pyvis.network import Network
import random
import jsonlines
from openai import OpenAI
import faiss
import time
# from vllm import LLM, SamplingParams

# 知识图谱抽取

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="2WikiMQA")
    parser.add_argument('--datapath', type=str, default='KD/teacher_kgc_data.json')
    parser.add_argument('--save_path', type=str, default='KD/teacher_kgc_res.json')
    parser.add_argument('--encoder', type=str, default='../bge-large-en-v1.5')
    parser.add_argument('--use_api', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--model_path', type=str, default='../Qwen2-7B-Instruct')
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--use_vllm', type=bool, default=False) 
    parser.add_argument('--mode', type=str, default="test")   
    args = parser.parse_args()
    
    with open(f"datasets/{args.dataset}/{args.datapath}") as f:
        dataset = json.load(f) 

    encoder = SentenceTransformer(args.encoder) 
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    if args.use_api:
        llm = OpenAI(
            api_key="sk-Bte8prbPVUtZMowI3e89D7F7A92743399fB5Bf98D509A9B7", 
            base_url="https://aihubmix.com/v1"
        )
        kgc = KGC(args.model_name, llm, None, encoder, args)
        kgcRefine = KGCRefine(args.model_name, llm, None, encoder, args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)    
        llm = LLM(model=args.model_path)
        kgc = KGC(None, llm, tokenizer, encoder, args)
        kgcRefine = KGCRefine(None, llm, tokenizer, encoder, args)
    
    
    def pipeline(i, relations, embeddingDB):
        question = dataset[i]["question"]  
        answer = dataset[i]["answer"]
        context = dataset[i]["context"]
        texts_all = []
        text2topic = {}
        for topic, texts in context:
            # texts = ' '.join(texts)
            texts_all += texts
            for text in texts:
                text2topic[text] = [topic]
        # try:
        triples_lists, relations = kgc.extract(text2topic, texts_all, relations, embeddingDB)
        time3 = time.time()
        entity_hint_list, relation_hint_list = kgcRefine.construct_refinement_hint(text2topic, texts_all, triples_lists,
                                                                                   relations)
        time4 = time.time()
        # print("construct_refinement_hint time: ", time4-time3)
        refined_triplets_list = []
 
        for idx, input_text in enumerate(texts_all):
            entity_hint_str = entity_hint_list[idx]
            relation_hint_str = relation_hint_list[idx]
            topic = text2topic[input_text][0]
            refined_triplets = kgcRefine.extractTriples(input_text, entity_hint_str, relation_hint_str)
            if not refined_triplets:
                refined_triplets = triples_lists[idx]
            refined_triplets_list.append(refined_triplets)
        time5 = time.time()
        # print("refine extractTriples time: ", time5-time4)
        return relations, embeddingDB, {"question": question, "answer": answer, "context": context, "triples": refined_triplets_list}

    save_path = f"datasets/{args.dataset}/{args.save_path}"
    if os.path.exists(save_path):
        with open(save_path) as f:
            start = len(f.readlines())
    else:
        start = 0

    if args.mode == "test":
        save_path = f"datasets/{args.dataset}/RAG/teacherLLM"
        with open(f"{save_path}/kgc.json", "a") as f:
            writer = jsonlines.Writer(f)
            for i, data in tqdm(enumerate(dataset[start:]), total=len(dataset[start:])):
                with open(f"{save_path}/relation.json") as f2:
                    relations = json.load(f2)
                embeddingDB = EmbeddingDatabase()
                if os.path.exists(f"{save_path}/relation_db.faiss"):
                    embeddingDB.load_database(f"{save_path}/relation_db")
                else:
                    if relations:
                        embeddingDB.create_index(list(relations.values()))
                
                relations, embeddingDB, new = pipeline(i+start, relations, embeddingDB)
                writer.write(new)
                with open(f"{save_path}/relation.json", "w") as f2:
                    json.dump(relations, f2)
                embeddingDB.save_database(f"{save_path}/relation_db")
                    

    else:
        with open(save_path, 'a') as f:
            writer = jsonlines.Writer(f)
            for i, data in tqdm(enumerate(dataset[start:]), total=len(dataset[start:])):
                # time1 = time.time()
                with open(f"datasets/{args.dataset}/KD/relation.json") as f2:
                    relations = json.load(f2)
                embeddingDB = EmbeddingDatabase()
                if os.path.exists(f"datasets/{args.dataset}/KD/relation_db.faiss"):
                    embeddingDB.load_database(f"datasets/{args.dataset}/KD/relation_db")
                else:
                    if relations:
                        embeddingDB.create_index(list(relations.values()))
                time2 = time.time()
                # print("准备时间: ", time2-time1)
                relations, embeddingDB, new = pipeline(i+start, relations, embeddingDB)
                writer.write(new)
                with open(f"datasets/{args.dataset}/KD/relation.json", "w") as f2:
                    json.dump(relations, f2)
                embeddingDB.save_database(f"datasets/{args.dataset}/KD/relation_db")
                try:
                    relations, embeddingDB, new = pipeline(i+start, relations, embeddingDB)
                    writer.write(new)
                    with open(f"datasets/{args.dataset}/KD/relation.json", "w") as f:
                        json.dump(relations, f)
                    embeddingDB.save_database(f"datasets/{args.dataset}/KD/relation_db")
            
                except:
                    writer.write({"wrong_index": start+i})
                    continue

        q2res = {}
        with open(f"datasets/{args.dataset}/{args.save_path}") as f:
            for line in f.readlines():
                new = json.loads(line)
                if "question" in new.keys():
                    q2res[new["question"]] = new
        
        wrong_idx = []

        res = []
        for i, item in enumerate(dataset):
            if item["question"] in q2res.keys():
                res.append(q2res[item["question"]])
            else:
                res.append({})
                wrong_idx.append(i)
        
        for idx in tqdm(wrong_idx):
            with open(f"datasets/{args.dataset}/KD/relation.json") as f2:
                    relations = json.load(f2)
            embeddingDB = EmbeddingDatabase()
            if os.path.exists(f"datasets/{args.dataset}/KD/relation_db.faiss"):
                embeddingDB.load_database(f"datasets/{args.dataset}/KD/relation_db")
            else:
                if relations:
                    embeddingDB.create_index(list(relations.values()))
            try:
                relations, embeddingDB, new = pipeline(idx, relations, embeddingDB)
                res[idx] = new
                with open(f"datasets/{args.dataset}/KD/relation.json", "w") as f:
                    json.dump(relations, f)
                embeddingDB.save_database(f"datasets/{args.dataset}/KD/relation_db")
            except:
                res[idx] = {"wrong_index": idx}
                continue

        with open(f"datasets/{args.dataset}/{args.save_path.replace('.json', '_.json')}", 'w') as f:
            for item in res:
                json.dump(item, f)
                f.write('\n')