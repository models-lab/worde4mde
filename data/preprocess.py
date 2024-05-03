import glob
import json
import logging
import re
import fasttext
import os
import pdftotext
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random

from modelset_evaluation.evaluation_classification_clustering import set_up_modelset
from w2v.w2v import MODELS, load_model

def read_pdf(file):
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f, raw=True)
    all_pdf = "\n".join(pdf)
    # trans-\nformation
    all_pdf = re.sub(r'-\n', '', all_pdf)
    # remove urls
    all_pdf = re.sub(r"http\S+", "", all_pdf)
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

    # tokenization by words
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
        try:
            content = read_pdf(f)
        except:
            logging.getLogger().info(f'Error in file {f}')
            continue
        tokens = preprocess_doc(content)
        result += tokens

    return result

def preprocess_sodump(args):
    with open("./stackoverflow.txt", 'r') as file:
        # Que categorias quiero
        lines = [l.strip() for l in file.readlines()]
        #print(lines)
        # Directorio donde estan todas las categorias
        directories = [d for d in os.listdir('/data2/sodump/all') if
                       os.path.isdir(os.path.join('/data2/sodump/all', d))]
        # Quiero su interseccion
        common_elements = set(lines) & set(directories)
        # Por cada directorio deseado me quedo con Posts y Comments
        dataset = []
        clean = re.compile('<.*?>')
        prob = args.sample_size / 100
        for element in common_elements:
            print("Preprocessing " + element)
            postpath = '/data2/sodump/all/' + element + '/Posts.xml'
            commentpath = '/data2/sodump/all/' + element + '/Comments.xml'

            posthistorypath = '/data2/sodump/all/' + element + '/PostHistory.xml'
            # Abro Posts
            # Events - signify when to yield a result
            events_ = ('start', 'end')  # Yield on the start and end of a tag
            events_ns = ('start', 'start-ns', 'end', 'end-ns')  # Yield on start and end of tags and namespaces
            campos_body = []
            campos_text = []
            if "Posts" in element:
                #with open(postpath, 'r') as posts:
                # Create an an iterable

                for event, e in ET.iterparse(postpath, events=events_):
                    if e.tag == "posts":
                        continue
                    if event == 'start':
                        body = e.get("Body")
                        re.sub(clean, '', body)
                        #print(body)
                        if body is None:
                            raise ValueError(str(e))
                        if random.random() < prob:
                            campos_body.append(body)
                    if event == 'end':
                        e.clear()
            elif "Comments" in element:
                for event, e in ET.iterparse(commentpath, events=events_):
                    if e.tag == "comments":
                        continue
                    if event == 'start':
                        text = e.get("Text")
                        re.sub(clean, '', text)
                        #print(text)
                        if text is None:
                            raise ValueError(str(e))
                        if random.random() < prob:
                            campos_text.append(text)
                    if event == 'end':
                        e.clear()

            #campos_body = random.sample(campos_body, round(len(campos_body) * (args.sample_size / 100)))
            #campos_text = random.sample(campos_text, round(len(campos_text) * (args.sample_size / 100)))
            dataset += campos_body + campos_text


    # Dataset contiene todo.
    tokenized_files = []
    print("LONGITUD DATASET")
    print(len(dataset))
    cnt = 0
    for content in dataset:
        # Eliminar tags.
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
        tokens = preprocess_doc(content)

        tokenized_files += tokens
    return tokenized_files

def load_data_metamodel_concepts(file_name):
    with open(file_name, "rb") as f:
        data = json.load(f)
    return data


def inside_vocabs(word, models):
    for (model, m) in models:
        if m == 'fasttext_bin' or m == 'fasttext-all':
            if word not in model.wv.key_to_index:
                return False
        else:
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
    files.sort()  # ensure reproducibility
    result = []
    for file_name in tqdm(files, desc='Preprocessing files'):
        data = load_data_metamodel_concepts(file_name)
        result += data

    models = []
    for m in MODELS:
        w2v_model = load_model(m, args.embeddings_out)
        models.append((w2v_model, m))
    result = [normalize_item(item, models) for item in result]
    result = [item for item in result if item["context"] is not None and item["recommendations"] != []]
    result = [item for item in result if item["context_type"] == args.context_type]
    modelset_df, _ = set_up_modelset(args)
    result = [item for item in result if item["id"] in list(modelset_df['id'])]
    return result
