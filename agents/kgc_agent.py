from langchain.agents import AgentExecutor, Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import json
import ast
import re
from prompts.HotpotQA import simple_extract_prompt
import random
from utils.embedding_database import EmbeddingDatabase

class Simple_KGC_Agent:
    def __init__(self, model, tokenizer, encoder, use_vllm=False):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.use_vllm = use_vllm
        self.relations = {}
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.embedding_database = EmbeddingDatabase(model_name='../bge-large-en-v1.5')

    def generating(self, prompt):
        """generate response from llm"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        if self.tokenizer:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if self.use_vllm:
                from vllm import SamplingParams
                sampling_params = SamplingParams(temperature=0.7, top_p=0.8, 
                                               repetition_penalty=1.2, max_tokens=2048)
                output = self.model.generate([text], sampling_params)
                return output[0].outputs[0].text
            else:
                model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda:0')
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
        else:
            response = self.model.chat.completions.create(
                messages=messages,
                model="gpt-3.5-turbo",
            )
            return response.choices[0].message.content

    def generating_batch(self, prompts):
        if self.tokenizer:
            batch_texts = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)
            
            if self.use_vllm:
                from vllm import SamplingParams
                sampling_params = SamplingParams(temperature=0.7, top_p=0.8, 
                                            repetition_penalty=1.2, max_tokens=2048)
                outputs = self.model.generate(batch_texts, sampling_params)
                return [output.outputs[0].text for output in outputs]
            else:
                model_inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to("cuda:0")
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                responses = []
                for i in range(len(generated_ids)):
                    output_ids = generated_ids[i][len(model_inputs.input_ids[i]):]
                    response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    responses.append(response)
                return responses
        else:
            responses = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                response = self.model.chat.completions.create(
                    messages=messages,
                    model="gpt-3.5-turbo",
                )
                responses.append(response.choices[0].message.content)
            return responses

    def extract(self, topic, text):
        prompt = simple_extract_prompt.prompt(topic, text)
        extracted_triples = self.generating(prompt).replace("`", "")
        
        try:
            res = ast.literal_eval(extracted_triples)
            return res
        except:
            res = []
            triples_ = extracted_triples[1:-1].split("], ")
            for tri in triples_:
                try:
                    tri_ = ast.literal_eval(tri + ']')
                    if len(tri_) == 3:
                        res.append(tri_)
                except:
                    tri = tri[1:].split(', ')
                    tri_ = []
                    for n in tri:
                        n = n[1:-1].replace("'", "")
                        tri_.append(n)
                    if len(tri_) == 3:
                        res.append(tri_)
            return res
    
    def extract_batch(self, topic_text_pairs):
        triples_lists = []
        prompts = [simple_extract_prompt.prompt(topic, text) for topic, text in topic_text_pairs]
        extracted_responses = self.generating_batch(prompts)
        all_triples = []
        for extracted_triples in extracted_responses:
            extracted_triples = extracted_triples.replace("`", "")
            try:
                res = ast.literal_eval(extracted_triples)
                all_triples.append(res)

            except:
                res = []
                if ":\n\n" in extracted_triples:
                    extracted_triples = extracted_triples.split(":\n\n")[1]
                if ":\n" in extracted_triples:
                    extracted_triples = extracted_triples.split(":\n")[1]
                # print('extracted_triples: ', extracted_triples)
                extracted_triples = extracted_triples.replace('\n', ", ")
                triples_ = extracted_triples[1:-1].split("], ")
                for tri in triples_:
                    try:
                        tri_ = ast.literal_eval(tri + ']')
                        res.append(tri_)
                    except:
                        tri = tri[1:].split(', ')
                        tri_ = []
                        for n in tri:
                            n = n[1:-1].replace("'", "")
                            tri_.append(n)
                        res.append(tri_)
                all_triples.append(res)
        # print("all triples: ", all_triples)
        return all_triples                