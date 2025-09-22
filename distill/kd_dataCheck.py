import json
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
from tqdm import tqdm
import random

encoder = SentenceTransformer('../bge-large-en-v1.5').to('cuda')

def completenessCheck(triples):
    for triple in triples:
        if len(triple) != 3:
            return 0
        for item in triple:
            if type(item) != str or item.lower() == 'none':
                return 0
    return 1

def accuracyCheck(text, triples):

    relations, entities = set(), set()
    for triple in triples:
        relations.add(triple[1])
        entities.add(triple[0])
        entities.add(triple[2])
    
    relations = list(relations)
    entities = list(entities)

    # 规则1：检查所有元素之间是否存在语义重复
    def check_rule1(elements):
        if len(elements) < 2:
            return True
        
        embeddings = encoder.encode(elements, convert_to_tensor=True, device='cuda')
        
        # 计算余弦相似度矩阵
        cos_sim = torch.mm(embeddings, embeddings.t())
        n = len(elements)
        
        # 创建上三角掩码（排除对角线）
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device='cuda'), diagonal=1)

        # 提取上三角部分并检查是否有任意值 > 0.8
        if torch.any(cos_sim[mask] > 0.8):
            return False
        return True

    # 规则2：检查元素是否在文本中存在或语义相近
    def check_rule2(elements, text):
        # 预处理文本：转为小写并移除多余空格
        clean_text = re.sub(r'\s+', ' ', text.lower()).strip()
        text_tokens = clean_text.split()
        
        # 生成n-gram (n=1,2,3)
        text_ngrams = set()
        for n in range(1, 4):
            for i in range(len(text_tokens) - n + 1):
                ngram = ' '.join(text_tokens[i:i+n])
                text_ngrams.add(ngram)
        # print("text_ngrams: ", text_ngrams)
        # 检查每个元素
        elements_embeddings = encoder.encode(elements, convert_to_tensor=True, device='cuda')
        n_grams_embeddings = encoder.encode(list(text_ngrams), convert_to_tensor=True, device='cuda')

        cos_sim = torch.mm(elements_embeddings, n_grams_embeddings.t())

        elem_has_match = torch.any(cos_sim > 0.7, dim=1)

        # 所有元素都有匹配则返回1
        if torch.all(elem_has_match):
            return 1
        return 0


    # try:
    # 如果没有元素，直接通过
    if not relations or not entities:
        return 1
        
    # 检查规则1（无重复元素）
    if not check_rule1(relations):
        return 0
        
    if not check_rule1(entities):
        return 0

    # 检查规则2（元素存在于文本）
    if not check_rule2(relations, text):
        return 0
    if not check_rule2(entities, text):
        return 0   

    return 1
        
    # except Exception as e:
    #     print(f"处理错误: {e}")
    #     return 0


def checking(texts_data, triples_data):
    """检查数据的完整性和准确性"""
# （1）完整性规则1：每个步骤的生成结果都是按照指定格式进行输出 ——> 有问题的数据直接未保存,因此输入的数据已满足该规则
# （2）完整性规则2：最终抽取结果中的每个三元组都完整的包含头实体、关系和尾实体；
# （3）准确性规则1：最终抽取结果中的每个元素都不存在与其语义非常相似的另一个元素；
# （4）准确性规则2：抽取结果的三元组中的元素都存在于输入文本中或者有语义相近的词。
    
    filted_dataset = []

    for (texts, triples) in tqdm(zip(texts_data, triples_data),total=len(texts_data), desc="Checking data"):
        if "question" not in triples.keys():
            continue
        ans = triples["answer"]
        question = triples["question"]
        triples_list = triples["triples"]
        idx = 0
        for topic, context in texts["context"]:
            for chunk in context:
                try:
                    if completenessCheck(triples_list[idx]):
                        if accuracyCheck(chunk, triples_list[idx]):
                            filted_dataset.append({"topic": topic, "text": chunk, "triples": str(triples_list[idx])})
                    idx += 1
                except:
                    idx += 1
                    continue

    return filted_dataset


if __name__ == "__main__":
    datasets = ["HotpotQA", "2WikiMQA"]

    for dataset in datasets:
        filtered_dataset = []
        triples_data = []
        with open(f"datasets/{dataset}/KD/teacher_kgc_data.json") as f:
            texts_data = json.load(f)
        with open(f"datasets/{dataset}/KD/teacher_kgc_res.json") as f:
            for line in f.readlines():
                triples_data.append(json.loads(line))
        filtered_dataset = checking(texts_data, triples_data)

        with open(f"datasets/{dataset}/KD/filtered_kgc_data.json", "w") as f:   
            json.dump(filtered_dataset, f, indent=2)

        random.shuffle(filtered_dataset)
        train = filtered_dataset[: int(len(filtered_dataset)*0.9)]
        dev = filtered_dataset[int(len(filtered_dataset)*0.9): ]

        with open(f"datasets/{dataset}/KD/train.json", "w") as f:
            json.dump(train, f, indent=2)
        with open(f"datasets/{dataset}/KD/dev.json", "w") as f:
            json.dump(dev, f, indent=2)
    