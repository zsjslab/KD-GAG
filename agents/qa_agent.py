from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import re
from vllm import LLM, SamplingParams


class QA_Agent:
    def __init__(self, model, tokenizer, args, STR=False):
        self.model = model
        self.tokenizer = tokenizer
        self.STR = STR
        try:
            self.use_vllm = args.use_vllm
            self.device = args.device
        except:
            self.use_vllm = False
            self.device = 'cuda:0'

    def generating(self, prompt):

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
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.2, max_tokens=2048)
            output = self.model.generate([text], sampling_params)
            return output[0].outputs[0].text
            # response = self.model.chat.completions.create(
            #     model="Qwen2.5-7B-Instruct",
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.7
            #     )
            # return response.choices[0].message.content
        else:
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
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response

    def grag_answering(self, question, pruned_subgraph):
        prompt = f"""Please answer the question based on the knowledge from the following triplets, provide only the answer, and do not include any explanation or apologies. If you cannot infer the answer to the question from the provided triples, output 'unknown':
        Question: {question}
        Triplets: {pruned_subgraph}
        Answer: """
        return self.generating(prompt)

    def rag_answering(self, question, texts):
        contexts = ""
        for i, text in enumerate(texts):
            contexts += f"{i+1}. {text}\n"
        prompt = f"""Please answer the question based on the knowledge from the following texts, provide only the answer words, and do not include any explanation or apologies:
        Question: {question}
        Texts: {contexts}
        Answer: """
        ans = self.generating(prompt)
        return ans

    def grag_answering_batch(self, questions, pruned_subgraphs):
        prompts = []
        for question, subgraph in zip(questions, pruned_subgraphs):
            prompt = f"""Please answer the question based on the knowledge from the following triplets, provide only the answer, and do not include any explanation or apologies. If you cannot infer the answer to the question from the provided triples, output 'unknown':
        Question: {question}
        Triplets: {pruned_subgraph}
        Answer: """
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)
        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]