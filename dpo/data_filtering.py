import json
import random

datasets = ["HotpotQA", "2WikiMQA"]
train = []
for dataset in datasets:
    with open(f"datasets/{dataset}/DPO/_add_sampling_data_{i}.json") as f:
        for line in f.readlines():
            new = json.loads(line)
            if "wrong" not in new.keys():
                q = new["prompt"].split('\n')[1].split("Question: ")[1]
                if not new["chosen"]:
                    continue
                if type(new["chosen"][0]) == dict:
                    new["chosen"] = new["chosen"][1]["content"]
                    new["rejected"] = new["rejected"][1]["content"]
                if (new["max_score"] - new["min_score"] > 0.2) and (new["max_score"] > 0.5):
                    train.append({"prompt": new["prompt"], "chosen": str(new["chosen"]), "rejected": str(new["rejected"])})

random.shuffle(train)
for dataset in datasets:
    with open(f"datasets/{dataset}/DPO/train_mix.json", "w") as f:
        json.dump(train[:int(len(train)*0.9)], f)
    
    with open(f"datasets/{dataset}/DPO/dev_mix.json", "w") as f:
        json.dump(train[int(len(train)*0.9):], f)