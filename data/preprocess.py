import glob
import json
import re

import pdftotext
from modelset import load
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

from modelset_evaluation.evaluation_classification_clustering import get_multiset, tokenizer, get_duplicates
from w2v.w2v import MODELS, load_model


def read_pdf(file):
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f)
    all_pdf = "\n".join(pdf)
    return all_pdf


def preprocess_sentence(string):
    # lower case
    string = string.lower()

    # remove \n
    string = re.sub(r'\n', ' ', string).strip()

    # remove numbers
    string = re.sub(r'\d+', '', string)

    # remove punctuation
    string = re.sub(r'[^\w\s]', '', string).strip()

    # tokenization
    return word_tokenize(string)


def preprocess_doc(string):
    # remove \n
    string = re.sub(r'\n', ' ', string).strip()

    # split in sentences
    sentences = [preprocess_sentence(s) for s in sent_tokenize(string)]

    # remove short sentences
    return [s for s in sentences if len(s) > 5]


def preprocess_dataset(args):
    files = glob.glob(args.training_dataset + "/**/*.pdf", recursive=True)
    result = []
    for f in tqdm(files, desc='Preprocessing files'):
        content = read_pdf(f)
        tokens = preprocess_doc(content)
        result += tokens
    return result


def load_data_metamodel_concepts(file_name):
    with open(file_name, "rb") as f:
        data = json.load(f)
    return data


def inside_vocabs(word, models):
    for model in models:
        if word not in model.key_to_index:
            return False
    return True


def normalize_item(item, models):
    item_new = {"context": item["context"].lower(), "context_type": item["contextType"], "id": item["id"]}
    if not inside_vocabs(item["context"].lower(), models):
        item_new["context"] = None
    item_new["recommendations"] = [r.lower()
                                   for r in item["recommendations"]
                                   if inside_vocabs(r.lower(), models)]
    return item_new


def preprocess_dataset_metamodel_concepts(args):
    files = glob.glob(args.training_dataset_concepts + "/**/*.json", recursive=True)
    result = []
    for file_name in tqdm(files, desc='Preprocessing files'):
        data = load_data_metamodel_concepts(file_name)
        result += data

    models = []
    for m in MODELS:
        if m == 'word2vec-mde':
            w2v_model = load_model(m, args.embeddings_out)
        else:
            w2v_model = load_model(m)
        models.append(w2v_model)
    result = [normalize_item(item, models) for item in result]
    result = [item for item in result if item["context"] is not None and item["recommendations"] != []]
    if args.remove_duplicates:
        dataset = load(modeltype=args.model_type, selected_analysis=['stats'])
        modelset_df = dataset.to_normalized_df(min_occurrences_per_category=0)
        ids = list(modelset_df['id'])
        corpus = [dataset.as_txt(i) for i in ids]
        corpus_multiset = [get_multiset(tokenizer(doc)) for doc in corpus]
        representatives = list(get_duplicates(corpus_multiset, ids, args.t0, args.t1).keys())
        result = [item for item in result if item["id"] in representatives]
    return result
