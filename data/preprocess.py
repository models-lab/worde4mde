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
    with open("./selection.txt", 'r') as file:
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
        for element in common_elements:
            print("Preprocessing " + element)
            postpath = '/data2/sodump/all/' + element + '/Posts.xml'
            commentpath = '/data2/sodump/all/' + element + '/Comments.xml'
            posthistorypath = '/data2/sodump/all/' + element + '/PostHistory.xml'
            # Abro Posts
            with open(postpath, 'r') as posts:
                # Leo las lineas de Posts.
                tree = ET.parse(posts)
                root = tree.getroot()
                campos_body = []
                for e in root.findall(".//row"):
                    body = e.get("Body")
                    if body is None:
                        raise ValueError(str(e))
                    campos_body.append(body)
            with open(commentpath, 'r') as comments:
                # Leo las lineas de Comments.
                tree = ET.parse(comments)
                root = tree.getroot()
                campos_text = []
                for e in root.findall(".//row"):
                    text = e.get("Text")
                    if text is None:
                        raise ValueError(str(e))
                    campos_text.append(text)
            #with open(posthistorypath, 'r') as comments:
            #    # Leo las lineas de PostHistory.
            #    tree = ET.parse(comments)
            #    root = tree.getroot()
            #    campos_text2 = []
            #    for e in root.findall(".//row"):
            #        text = e.get("Text")
            #        if text is None:
            #            print(ET.tostring(e))
            #            raise ValueError(str(e))

            #        campos_text2.append(text)


            dataset += campos_body + campos_text


    # Dataset contiene todo.
    tokenized_files = []
    print("LONGITUD DATASET")
    print(len(dataset))
    cnt = 0
    for content in dataset:
        # Eliminar tags.
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
            print(content)
        tokens = preprocess_doc(content)

        tokenized_files += tokens
    return tokenized_files

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
    files.sort()  # ensure reproducibility
    result = []
    for file_name in tqdm(files, desc='Preprocessing files'):
        data = load_data_metamodel_concepts(file_name)
        result += data

    models = []
    for m in MODELS:
        if m != 'so_word2vec' and m != 'fasttext' and m!= 'skip_gram-mde':
            continue
        w2v_model = load_model(m, args.embeddings_out)
        models.append(w2v_model)
    result = [normalize_item(item, models) for item in result]
    result = [item for item in result if item["context"] is not None and item["recommendations"] != []]
    result = [item for item in result if item["context_type"] == args.context_type]
    modelset_df, _ = set_up_modelset(args)
    result = [item for item in result if item["id"] in list(modelset_df['id'])]
    return result
