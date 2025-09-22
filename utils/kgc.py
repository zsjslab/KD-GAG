from transformers import AutoModelForCausalLM, AutoTokenizer
# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tqdm import tqdm
import json
# from prompts.WikiMQA import extract_prompt, schema_prompt, sc_prompt
# from prompts.HotpotQA import extract_prompt, schema_prompt, sc_prompt
import ast
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from vllm import LLM, SamplingParams
import time

class KGC:

    def __init__(self, model_name, model, tokenizer, encoder, args):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.use_vllm = args.use_vllm
        if args.dataset == "2WikiMQA":
            from prompts.WikiMQA import extract_prompt, schema_prompt, sc_prompt
            self.extract_prompt = extract_prompt
            self.schema_prompt = schema_prompt
            self.sc_prompt = sc_prompt
        else:
            from prompts.HotpotQA import extract_prompt, schema_prompt, sc_prompt
            self.extract_prompt = extract_prompt
            self.schema_prompt = schema_prompt
            self.sc_prompt = sc_prompt

    def generating(self, prompt):

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
                sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.2, max_tokens=2048)
                output = self.model.generate([text], sampling_params)
                return output[0].outputs[0].text
            else:
                model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

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
                model = self.model_name,
            )
            return response.choices[0].message.content


    def extractBase(self, topic, text):
        prompt = self.extract_prompt.prompt(topic, text)
        extracted_triples = self.generating(prompt).replace("`", "")
        try:
            res = ast.literal_eval(extracted_triples)
            return res
        except:
            res = []
            triples_ = extracted_triples[1:-1].split("], ")
            for tri in triples_:
                try:
                    res.append(ast.literal_eval(tri + ']'))
                except:
                    tri = tri[1:].split(', ')
                    tri_ = []
                    for n in tri:
                        n = n[1:-1].replace("'", "")
                        tri_.append(n)

                    res.append(tri_)
            return res

    def schemaDefine(self, text, extracted_triples, extracted_relations):
        prompt = self.schema_prompt.prompt(text, extracted_triples, extracted_relations)
        relations_define = self.generating(prompt)
        return relations_define

    def schemaCanonicalization(self, text, extracted_triples, extracted_relation, relation_define, choices):
        prompt = self.sc_prompt.prompt(text, extracted_triples, extracted_relation, relation_define, choices)
        # print("!!!prompt: ", prompt)
        choice = self.generating(prompt)
        # print("!!!choice: ", choice)
        return choice

    def findRel(self, relations, define):
        rel = [key for key, value in relations.items() if value.lower() == define]
        if rel:
            return rel[0]
        else:
            print("define: ", define)

    def extract(self, text2topic_idx, texts, relations, embeddingDB):
        triples = []
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, text in enumerate(texts):
            topic = text2topic_idx[text][0]
            extracted_triples = self.extractBase(topic, text)
            extracted_triples_ = []
            for triple in extracted_triples:
                if len(triple) == 3:
                    extracted_triples_.append(triple)
            extracted_relations = []
            relations_define = {}
            schema_triples = []
            for tri in extracted_triples_:
                if tri[1] in relations.keys():
                    relations_define[tri[1].lower()] = relations[tri[1].lower()]
                    continue
                extracted_relations.append(tri[1])
                schema_triples.append(tri)
            relations_define_res = self.schemaDefine(text, schema_triples, extracted_relations)
            # print("relations_define_res: ", relations_define_res)
            relations_define_res = relations_define_res.replace('：', ': ').replace("`", "").replace('Answer:\n', '').replace('- ', '').replace("**", '').split('\n')
            # print("relations_define: ", relations_define_res)

            for rel_define in relations_define_res:
                if rel_define and ':' in rel_define:
                    try:
                        rel = ast.literal_eval(rel_define.split(': ')[0]).lower()
                    except:
                        rel = rel_define.split(': ')[0].lower()
                    # print("rel_define: ", rel_define)
                    try:
                        define = rel_define.split(': ')[1].lower()
                        relations_define[rel] = define
                    except:
                        continue

            if not relations:
                for rel, define in relations_define.items():
                    relations[rel] = define
                    embeddingDB.add_strings([define])
            # else:
            #     print("num of relations: ", len(relations.keys()))

                for idx in range(len(extracted_triples_)):
                    triple = extracted_triples_[idx]
                    # print("relation: ", triple[1])
                    if triple[1].lower() in relations.keys():
                        continue
                    choices = ""

                    time1 = time.time()
                    items = embeddingDB.search_similar(triple[1], 10)
                    time2 = time.time()
                    # print("search time: ", time2-time1)
                    if items:
                        print("similar: ", items)
                        for c_idx, item in enumerate(items):
                            define = item[0]
                            rel = self.findRel(relations, define.lower())
                            choices += alphabet[c_idx] + ". '" + rel + "': " + define + '\n' 

                        choices += alphabet[c_idx + 1] + ". None of the above.\n"
                        choose = self.schemaCanonicalization(text, triple, triple[1], relations_define[triple[1].lower()], choices)
                        try:
                            choice = choose.split('.')[0]
                            print("relation: ", triple[1])
                            print("choices: ", choices)
                            print("choice: ", choice)
                        except:
                            print("relation: ", triple[1], "  relations defined: ", relations_define.keys())
                            print("choice: ", choose)
                        
                        if alphabet[c_idx+1] == choice:
                            relations[triple[1].lower()] = relations_define[triple[1].lower()]
                            embeddingDB.add_strings([relations_define[triple[1].lower()]])
                        else:
                            pattern = r"^" + choice + r"\. '(.*?)'"
                            result = re.search(pattern, choices, re.MULTILINE).group(1)
                            # print(triple[1], '\n', choices, choice, '\n')
                            extracted_triples_[idx][1] = result
                    else:
                        try:
                            relations[triple[1].lower()] = relations_define[triple[1].lower()]
                            embeddingDB.add_strings([relations_define[triple[1].lower()]])
                        except:
                            continue
                    time4 = time.time()
                    # print("schema canonicalization time: ", time4 - time2)
            # res.append({"text": text, "prediction": extracted_triples, "reference": reference})
            triples.append(extracted_triples_)
        return triples, relations