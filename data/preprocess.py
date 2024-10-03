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
from datasets import load_dataset
import numpy as np

from modelset_evaluation.evaluation_classification_clustering import set_up_modelset
from w2v.w2v import MODELS, load_model
from collections import namedtuple, Counter

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

def consider_post(allowed_ids, post_id, parent_id, post_tags, wanted_tags):    
    if wanted_tags is None:
        return True

    if post_tags is None:
        if parent_id in allowed_ids:
            print("Selected dependent: ", post_id)
            allowed_ids.add(post_id)
            return True
    else:
        # Example: "&lt;oop&gt;&lt;uml&gt;&lt;associations&gt;&lt;aggregation&gt;&lt;composition&gt;"
        tag_list = post_tags.replace('<', '').split('>')[:-1]
        if False and len(tag_list) > 0:
            print(tag_list)
            print(wanted_tags)
            
        for tag in tag_list:
            for wanted in wanted_tags:
                if wanted in tag.lower():
                    allowed_ids.add(post_id)
                    print("Selected ", post_id, " ", tag_list)
                    return True
        return False

Stats = namedtuple("Stats", ["num_posts", "num_sentences", "num_tokens",
                             "num_unique_words", "unique_words", "avg_sentence_length", "std_sentence_length", "tag_counter"])

def preprocess_sodump(args, selection_file, tags=None, use_comments=True):
    fname = args
    wanted_tags = None
    allowed_ids = set()
    if tags is not None:
        wanted_tags = [s.strip().lower() for s in tags.split(',')]
        logging.getLogger().info(f'Training with tags: ' + ",".join(wanted_tags))

    with open(selection_file, 'r') as file:
        # Que categorias quiero
        lines = [l.strip() for l in file.readlines()]
        #print(lines)
        # Directorio donde estan todas las categorias
#        directories = [d for d in os.listdir('/data2/sodump/all') if
#                       os.path.isdir(os.path.join('/data2/sodump/all', d))]
        # Quiero su interseccion
#        common_elements = set(lines) & set(directories)
        # Por cada directorio deseado me quedo con Posts y Comments
        dataset = []
        clean = re.compile('<.*?>')
        prob = args.sample_size / 100

        common_elements = lines
        print(common_elements)
        
#        common_elements = ['stackoverflow.com-Posts', 'stackoverflow.com-Comments']
                           
        number_posts = 0
        tag_counter = Counter()

        for element in common_elements:
            print("Preprocessing " + element)
            print("Current length ", len(dataset))
            
            postpath = '/data2/sodump/all/' + element + '/Posts.xml'
            commentpath = '/data2/sodump/all/' + element + '/Comments.xml'

            posthistorypath = '/data2/sodump/all/' + element + '/PostHistory.xml'
            # Abro Posts
            # Events - signify when to yield a result
            events_ = ('start', 'end')  # Yield on the start and end of a tag
            events_ns = ('start', 'start-ns', 'end', 'end-ns')  # Yield on start and end of tags and namespaces
            campos_body = []
            campos_text = []
            
            if os.path.isfile(postpath):
                #with open(postpath, 'r') as posts:
                # Create an an iterable
                print("Processing paths: ", postpath)
                for event, e in ET.iterparse(postpath, events=events_):
                    if e.tag == "posts":
                        continue
                    if event == 'start':
                        post_id = e.get("Id")
                        parent_id = e.get("ParentId")
                        post_tags = e.get("Tags")               
                        if consider_post(allowed_ids, post_id, parent_id, post_tags, wanted_tags):
                            body = e.get("Body")
                            re.sub(clean, '', body)
                            #print(body)
                            if body is None:
                                raise ValueError(str(e))
                            if wanted_tags is not None or random.random() < prob:
                                campos_body.append(body)
                                number_posts += 1

                                if post_tags is not None:
                                    tag_list = post_tags.replace('<', '').split('>')[:-1]
                                    for t in tag_list:
                                        tag_counter[t] += 1


                    if event == 'end':
                        e.clear()
            if use_comments and os.path.isfile(commentpath):
                print("Processing comments", commentpath)
                for event, e in ET.iterparse(commentpath, events=events_):
                    if e.tag == "comments":
                        continue
                    if event == 'start':
                        post_id = e.get("PostId")
                        if wanted_tags is None or post_id in allowed_ids:
                            text = e.get("Text")
                            re.sub(clean, '', text)
                            #print(text)
                            if text is None:
                                raise ValueError(str(e))
                            if wanted_tags is not None or random.random() < prob:
                                campos_text.append(text)
                                number_posts += 1

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
    token_counter = Counter()
    for content in dataset:
        # Eliminar tags.
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)

        tokens = preprocess_doc(content)
        tokenized_files += tokens

        for line in tokens:
            for t in line:
                token_counter[t] += 1

    number_of_sentences = len(tokenized_files)
    sent_lens = [len(sentence) for sentence in tokenized_files]
    number_tokens = sum(sent_lens)

    unique_words = len(token_counter)
    # most_common_words = token_counter(text, num_words=50)
    average_sentence_length = np.mean(sent_lens)
    std_sentence_length = np.std(sent_lens)

    stats = Stats(number_posts, number_of_sentences, number_tokens, unique_words, token_counter, average_sentence_length, std_sentence_length, tag_counter)

    return tokenized_files, stats

def preprocess_wikipedia(args):
    # Load HuggingFace wikipedia
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    tokenized_files = []
    for split_name, split_dataset in tqdm(wikipedia_dataset.items(), desc='Preprocessing wiki splits'):
        for row in tqdm(split_dataset, desc='Preprocessing wiki files'):
            tokenized_files += preprocess_doc(row["text"][:int(len(row["text"]) * 0.15)])
    return tokenized_files

def load_data_metamodel_concepts(file_name):
    with open(file_name, "rb") as f:
        data = json.load(f)
    return data


def inside_vocabs(word, models):
    for (model, m) in models:
        if m == 'fasttext_bin' or m == 'fasttext-all' or m == 'stackoverflow_modeling' or m == 'fasttext_wikipedia_modelling':
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
