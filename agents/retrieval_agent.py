from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import ast
import re
from vllm import LLM, SamplingParams
import os
import sys
from time import time

# 获取当前文件的绝对路径，然后找到其父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 parent_dir 添加到 Python 的模块搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.embedding_database import EmbeddingDatabase


class Retrieval_Agent:
    def __init__(self, encoder, model, ft_model, tokenizer, args, prune_mode='test'):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        try:
            self.device = args.device
            self.use_vllm = args.use_vllm
        except:
            self.device = 'cuda:0'
            self.use_vllm = False
        self.prune_mode = prune_mode
        # if args.dataset == "HotpotQA":
        from prompts.HotpotQA import topic_entity_prompt
        self.topic_entity_prompt = topic_entity_prompt
        # elif args.dataset == "2WikiMQA":
            # from prompts.WikiMQA import topic_entity_prompt
            # self.topic_entity_prompt = topic_entity_prompt
        self.ft_model = ft_model
        self.graph = None
    
    def set_graph(self, graph, context=None):
        kg = nx.DiGraph()
        entities, relations = set(), set()
        if context:
            self.triple2text = {}
            k = 0
            for topic, texts in context:
                for text in texts:
                    for triple in graph[k]:
                        self.triple2text[str(triple)] = text
                    k += 1

        # if type(graph[0][0]) != str:
        for triples in graph:
            if not triples:
                continue
            for triple in triples:
                try:
                    kg.add_edge(str(triple[0]), str(triple[2]), relation=triple[1])
                    entities.add(str(triple[0]))
                    entities.add(str(triple[2]))
                    relations.add(triple[1])
                except:
                    print("triples: ", triple, '\n')
        # else:
        #     for triple in graph:
        #         if not triple:
        #             continue
        #         try:
        #             kg.add_edge(str(triple[0]), str(triple[2]), relation=triple[1])
        #             entities.add(str(triple[0]))
        #             entities.add(str(triple[2]))
        #             relations.add(triple[1])
        #         except:
        #             print(triple, '\n')

        self.graph = kg
        self.entities = entities
        self.relations = relations
        self.entity_db = EmbeddingDatabase()
        self.entity_db.create_index(list(entities))
        

    
    def generating(self, prompt, task, mode='normal', temperature=0.7):

        if self.use_vllm:
            messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if mode == 'normal':
                sampling_params = SamplingParams(temperature=temperature, repetition_penalty=1.2, max_tokens=2048)
                output = self.model.generate([text], sampling_params)
                return output[0].outputs[0].text
            else:
                responses = []
                sampling_params = SamplingParams(temperature=temperature, repetition_penalty=1.2, max_tokens=2048)
                for _ in range(5):
                    output = self.model.generate([text], sampling_params)
                    responses.append(output[0].outputs[0].text)
                return responses

        if task == 'prune':
            messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.ft_model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            if mode == 'normal':
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
            else:
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=5,
                    # top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                responses = []
                for res_ids in generated_ids:
                    res_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, [res_ids])
                    ]
                    response = self.tokenizer.batch_decode(res_ids, skip_special_tokens=True)[0]
                    responses.append(response)
                return responses


    def TopicEntity(self, question):
        prompt = self.topic_entity_prompt.prompt(question)
        preds = self.generating(prompt, task='te')
        try:
            preds = ast.literal_eval(preds)
        except:
            preds_ = []
            for pred in preds[1:-1].split(', '):
                preds_.append(pred[1:-1])
            preds = preds_
        tes = []
        for te in preds:
            topic_entities = self.entity_db.search_similar(te, top_k=3, min_similarity=0.5)
            for topic_entity, _ in topic_entities:
                if topic_entity not in tes:
                    tes.append(topic_entity)
        return preds, tes

    def SubgraphRetrieval(self, topic_entities, hop=3):
        subgraph = []
        visited = set()
        for te in topic_entities:
            centers = [(te, 0)]

            triples = []
            while centers:
                center_node, depth = centers.pop(0)
                # print("center_node: ", center_node)
                visited.add(center_node)
                if depth < hop:
                    neighbors_r = list(self.graph.neighbors(center_node))
                    neighbors_l = list(self.graph.predecessors(center_node))
                    # print('neighbors: ', neighbors)
                    for neighbor in neighbors_r:
                        if neighbor in visited:
                            continue
                        relation = self.graph.get_edge_data(center_node, neighbor)['relation']
                        if [neighbor, relation, center_node] not in triples:
                            triples.append([center_node, relation, neighbor])
                        centers.append((neighbor, depth + 1))
                    for neighbor in neighbors_l:
                        if neighbor in visited:
                            continue
                        relation = self.graph.get_edge_data(neighbor, center_node)['relation']
                        if [center_node, relation, neighbor] not in triples:
                            triples.append([neighbor, relation, center_node])
                        centers.append((neighbor, depth + 1))
            subgraph += triples
        return subgraph

    def Prune(self, extract_entities, subgraph, question, temperature=0.7):
        prompt = f"""You are provided with a list of triples in the format [subject, relation, object]. Your task is to extract and return only the triples that are relevant to answering the following question. The extracted triples must come exactly from the provided list without any modification—do not change, add, or remove any words or punctuation. Only include triples that contribute direct or indirect support for answering the question. Output nothing but the relevant triple(s) in their original format and do not include any explanation or apologies.
        
        Question: {question}
        Triples: {subgraph}
        Relevant triples:"""
        
        pruned_triples_res = self.generating(prompt, task='prune', mode='normal', temperature=temperature)
        pruned_list = []
        if type(pruned_triples_res) != list:
            pruned_triples_res = [pruned_triples_res]

        for pruned_triples in pruned_triples_res:
            try:
                pruned_subgraph = ast.literal_eval(pruned_triples)   
            except:
                try:
                    pruned_subgraph = ast.literal_eval(pruned_triples.replace('\n', ', ')) 
                except:
                    pruned_subgraph = []
                    pattern = r'\[(.*?)\]' 
                    output_triples = re.findall(pattern, pruned_triples)
                    for text in output_triples:
                        pruned_subgraph.append(text.split(', '))
            pruned_list.append(pruned_subgraph)
        return prompt, pruned_list

    def textRetrieve(self, triples):
        texts = {}
        for tri in triples:
            try:
                text = self.triple2text[str(tri)]
            except:
                continue
            if text not in texts:
                texts[text] = 0
            texts[text] += 1
        texts_ = sorted(texts.items(), key=lambda x: x[1], reverse=True)[:4]

        knowledge_texts = []
        for text, _ in texts_:
            knowledge_texts.append(text)
        return knowledge_texts

    def graphRetrieve(self, triples, context, question, temperature=0.7):
        if self.prune_mode != 'sampling':
            self.set_graph(triples, context)
            _, tes = self.TopicEntity(question)
            subgraph = self.SubgraphRetrieval(tes)
            pruned_prompt, pruned_subgraph = self.Prune(tes, subgraph, question, temperature)
            return pruned_prompt, pruned_subgraph
        else:
            if not self.graph or triples != self.triples:
                self.set_graph(triples)
                _, self.tes = self.TopicEntity(question)
                self.subgraph = self.SubgraphRetrieval(self.tes)

            pruned_prompt, pruned_subgraph = self.Prune(self.tes, self.subgraph, question, temperature)
            self.triples = triples
            return pruned_prompt, pruned_subgraph
    
    def grag_answering(self, question, pruned_subgraph, STR=True):
        prompt = f"""Please answer the question based on the knowledge from the following triplets, provide only the answer, and do not include any explanation or apologies. If you cannot infer the answer to the question from the provided triples, output 'unknown':
        Question: {question}
        Triplets: {pruned_subgraph}
        Answer: """

        ans = self.generating(prompt, task='qa')
        return ans

    def clear_graph(self):
        self.graph = None
        self.entities = None
        self.relations = None
        self.entity_db = None
        self.triples = None
        self.tes = None
        self.triple2text = None
        if hasattr(self, 'entity_db'):
            del self.entity_db
        