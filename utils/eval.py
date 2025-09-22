import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import pickle as pkl
import itertools
import argparse
import ast

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    # print(prediction, ground_truth)
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HotpotQA')
    parser.add_argument('--version', type=str, default='responses')

    args = parser.parse_args()

    res = []
    with open(f"datasets/{args.dataset}/RAG/{args.version}") as f:
        for line in f.readlines():
            res.append(json.loads(line))

    filter_res = [data for data in res if 'pred' in data.keys()]

    f1s, emss, accs = [], [], []

    for r in filter_res:
        try:
            r['pred'] = str(ast.literal_eval(r['pred']))
        except:
            pass
        r['answer'] = r['answer'].lower().replace("_", " ").replace("-", " ").replace(",", "") in r['pred'].lower().replace("_", " ").replace("-", " ").replace(",", "")
        r['pred'] = r['pred'].lower().replace("_", " ").replace("-", " ").replace(",", "") in r['answer'].lower().replace("_", " ").replace("-", " ").replace(",", "")
        f1s.append(f1_score(r['pred'], r['answer']))
        emss.append(exact_match_score(r['pred'], r['answer']))

    print('F1:', np.mean(f1s))
    print('EM:', np.mean(emss))