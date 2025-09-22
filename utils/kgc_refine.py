from sentence_transformers.util import cos_sim
import re
import ast
import random
# from prompts.WikiMQA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
# from prompts.HotpotQA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
# from vllm import LLM, SamplingParams

class KGCRefine:
    def __init__(self, model_name, model, tokenizer, encoder, args):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.use_vllm = args.use_vllm
        if args.dataset == "2WikiMQA":
            from prompts.WikiMQA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
            self.entity_extract_prompt = entity_extract_prompt
            self.entity_merge_prompt = entity_merge_prompt
            self.extract_optim_prompt = extract_optim_prompt
        else:
            from prompts.HotpotQA import entity_extract_prompt, entity_merge_prompt, extract_optim_prompt
            self.entity_extract_prompt = entity_extract_prompt
            self.entity_merge_prompt = entity_merge_prompt
            self.extract_optim_prompt = extract_optim_prompt

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


    def entityExtract(self, topic, text):
        prompt = self.entity_extract_prompt.prompt(topic, text)
        entities = self.generating(prompt)
        try:
            entities = ast.literal_eval(entities)
        except:
            try:
                entities = list(re.search(r'\[(.*?)\]', entities).group(1).replace("'", "").split(', '))
            except:
                try:
                    entities = entities.split('\n')
                except:
                    print("error___extract entities: ", entities, "text: ", text)
        return entities

    def entityMerge(self, text, previous_entities, extracted_entities):
        prompt = self.entity_merge_prompt.prompt(text, previous_entities, extracted_entities)
        merged_entities = self.generating(prompt)
        if "[" not in merged_entities:
            merged_entities = merged_entities.split(', ')
            return merged_entities
        try:
            start_index = merged_entities.find('[')
            end_index = merged_entities.rfind(']')
            merged_entities = merged_entities[start_index:end_index+1]
            merged_entities = ast.literal_eval(merged_entities)
        except:
            print("error__merged entities: ", merged_entities, "text: ", text)
        return merged_entities

    def relevantRelations(self, input_text_str, previous_relations, topk=10):
        text_embedding = self.encoder.encode(input_text_str)
        similarity_dic = {}
        for rel in previous_relations:
            rel_embedding =  self.encoder.encode(rel)
            sim = cos_sim(text_embedding, rel_embedding)
            if sim > 0.5:
                similarity_dic[rel] = sim
        if len(similarity_dic) < 10:
            relevant_relations = list(similarity_dic.keys())
        else:
            relevant_relations = [rel for rel, sim in
                                  sorted(similarity_dic.items(), key=lambda item: item[1], reverse=True)][:topk]
        return relevant_relations

    def extractTriples(self, input_text, entity_hint_str, relation_hint_str):
        prompt = self.extract_optim_prompt.prompt(input_text, entity_hint_str, relation_hint_str)
        triples = self.generating(prompt)
        try:
            triples = ast.literal_eval(triples)
            triples_ = []
            for triple in triples:
                if len(triple) == 3:
                    triples_.append(triple)
            return triples_
        except:
            res = []
            triples_ = triples[1:-1].split("], ")
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

    def construct_refinement_hint(self, text2topic_idx, input_text_list, extracted_triplets_list, relations, include_relation_example="self",
                                  relation_top_k=10):

        entity_hint_list = []
        relation_hint_list = []

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{"text": input_text_str, "triplet": triplet}]
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        # print("________start constructing hint lists______")
        for idx in range(len(input_text_list)):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                if len(triplet) == 3:
                    previous_entities.add(triplet[0])
                    previous_entities.add(triplet[2])
                    previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            topic = text2topic_idx[input_text_str][0]
            extracted_entities = self.entityExtract(topic, input_text_str)
            # print("(1) extrated entities: ", extracted_entities)

            merged_entities = self.entityMerge(
                input_text_str, previous_entities, extracted_entities
            )
            # print("(2) merged entities: ", merged_entities)

            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = self.relevantRelations(input_text_str, previous_relations)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in relations.keys():
                    continue

                relation_definition = relations[relation]

                candidate_relation_str += f"{relation_idx + 1}. {relation}: {relation_definition}\n"

                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        # candidate_relation_str += "Example: None.\n"
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(relation_example_dict[relation])
                        # for example in relation_example_dict[relation]:
                        #     if example["text"] != input_text_str:
                        #         selected_example = example
                        #         break
                        if selected_example is not None:
                            candidate_relation_str += f"Example: '{selected_example['triplet']}' can be extracted from '{selected_example['text']}'\n"
                            # candidate_relation_str += f"""例如,{selected_example['triplet']}可以从"{selected_example['text']}中提取"\n"""
                        else:
                            # candidate_relation_str += "Example: None.\n"
                            pass
            relation_hint_list.append(candidate_relation_str)
            # print("(3) relation hint list: ", relation_hint_list)
        return entity_hint_list, relation_hint_list