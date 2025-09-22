from agents.kgc_agent import Simple_KGC_Agent
from agents.retrieval_agent import Retrieval_Agent
from agents.qa_agent import QA_Agent
from utils.embedding_database import EmbeddingDatabase
from tqdm import tqdm
import json
import torch
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments,PreTrainedModel,AutoConfig)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import jsonlines
import os
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HotpotQA')
    parser.add_argument('--mode', type=str, default='test', help="test for runing all data in test set, item for demonstration")
    # parser.add_argument('--model', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--pruner', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--pruner_checkpoint', type=str, default='1500')
    parser.add_argument('--kgc_model', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--kgc_checkpoint', type=str, default='5500')
    parser.add_argument('--encoder_path', type=str, default='../bge-large-en-v1.5')
    parser.add_argument('--use_vllm', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    encoder = SentenceTransformer(args.encoder_path)

    # load test data
    with open(f"datasets/{args.dataset}/RAG/test.json") as f:
        test_set = json.load(f)
    
    if args.mode == 'kgc':
    
        # create agent
        kgc_llm = AutoModelForCausalLM.from_pretrained(f"../{args.kgc_model}", device_map=None)
        kgc_tokenizer = AutoTokenizer.from_pretrained(f"../{args.kgc_model}", trust_remote_code=True)
        kgc_llm = PeftModel.from_pretrained(kgc_llm, f"distill/ft_students/{args.dataset}_{args.kgc_model}/checkpoint-{args.kgc_checkpoint}", device_map=None)
        kgc_llm = kgc_llm.to(args.device)
        kgc_llm = kgc_llm.merge_and_unload()
        kgc_llm.eval()
        kgc_agent = Simple_KGC_Agent(kgc_llm, kgc_tokenizer, encoder)
        
        # check existing data
        if not os.path.exists(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}"):
            os.makedirs(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}")

        if os.path.exists(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}.json"):
            with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}.json") as f:
                start = len(f.readlines())
        else:
            start = 0
            

        with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}.json", "a") as f:
            writer = jsonlines.Writer(f)
            for data in tqdm(test_set[start:], total=len(test_set[start:])):
                context = data["context"]
                triples_list = []
                # topic_text_pairs = []
                for topic, texts in context:
                    for text in texts:
                        triples = kgc_agent.extract(topic, text)
                        triples_ = []
                        for tri in triples:
                            if (tri not in triples_) and (len(tri)==3):
                                triples_.append(tri)
                        triples_list.append(triples_)
                writer.write({"question": data["question"], "answer": data["answer"], "content": context, "triples": triples_list})

    elif args.mode == 'rag':

        # create agent
        llm = AutoModelForCausalLM.from_pretrained(f"../Qwen2.5-7B-Instruct", device_map='auto', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(f"../Qwen2.5-7B-Instruct", trust_remote_code=True)
        if args.pruner_checkpoint:
            prune_llm = PeftModel.from_pretrained(llm, f"dpo/mix_qwen_2/checkpoint-{args.pruner_checkpoint}", device_map='auto')
            prune_llm = prune_llm.merge_and_unload()
        else:
            prune_llm = llm
        prune_llm.eval()
        retrieval_agent = Retrieval_Agent(encoder, llm, prune_llm, tokenizer, args)
        qa_agent = QA_Agent(llm, tokenizer, args)

        # load kgc data
        kgc_data = []
        with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/kgc.json") as f:
            for line in f.readlines():
                kgc_data.append(json.loads(line))

        # check existing data
        if not os.path.exists(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}"):
            os.makedirs(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}")

        if os.path.exists(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_{args.pruner_checkpoint}.json"):
            with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_{args.pruner_checkpoint}.json") as f:
                start = len(f.readlines())
        else:
            start = 0
        if args.pruner_checkpoint:
            with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_{args.pruner_checkpoint}.json", "a") as f:
                writer = jsonlines.Writer(f)
                for i, data in tqdm(enumerate(test_set[start:]), total=len(test_set[start:])):
                    question = data["question"]
                    answer = data["answer"]
                    context = data["context"]
                    triples = kgc_data[start+i]["triples"]
                    _, pruned_subgraph = retrieval_agent.graphRetrieve(triples, context, question)
                    pred = qa_agent.grag_answering(question, pruned_subgraph) 
                    
                    if pred.lower() == "unknown":
                        knowledge_texts = retrieval_agent.textRetrieve(pruned_subgraph)
                        if not knowledge_texts:
                            knowledge_texts = retrieval_agent.textRetrieve(pruned_subgraph[0])
                        pred = qa_agent.rag_answering(question, knowledge_texts)
                        new = {"question": question, "answer": answer, "pred": pred, "subgraph": pruned_subgraph, "knowledge texts": knowledge_texts}
                    else:
                        new = {"question": question, "answer": answer, "pred": pred, "subgraph": pruned_subgraph, "knowledge texts": []}
                    writer.write(new)
                    del new
                    retrieval_agent.clear_graph()
                    torch.cuda.empty_cache() 
        else:
            if os.path.exists(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_no_dpo.json"):
                with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_no_dpo.json") as f:
                    start = len(f.readlines())
            else:
                start = 0
            with open(f"datasets/{args.dataset}/RAG/{args.kgc_model}_{args.pruner}/{args.mode}_no_dpo.json", "a") as f:
                writer = jsonlines.Writer(f)
                for i, data in tqdm(enumerate(test_set[start:]), total=len(test_set[start:])):
                    question = data["question"]
                    answer = data["answer"]
                    context = data["context"]
                    triples = kgc_data[start+i]["triples"]
                    _, pruned_subgraph = retrieval_agent.graphRetrieve(triples, context, question)
                    cot, pred = qa_agent.grag_answering(question, pruned_subgraph) 
                    if pred == "unknown":
                        knowledge_texts = retrieval_agent.textRetrieve(pruned_subgraph)
                        pred = qa_agent.rag_answering(question, knowledge_texts)
                        new = {"question": question, "answer": answer, "pred": pred, "subgraph": pruned_subgraph, "knowledge texts": knowledge_texts}
                    else:
                        new = {"question": question, "answer": answer, "pred": pred, "subgraph": pruned_subgraph, "cot": cot}
                    writer.write(new)
                    del new
                    retrieval_agent.clear_graph()
                    torch.cuda.empty_cache()   
