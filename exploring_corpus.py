import glob
import logging
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
import xml.etree.ElementTree as ET

import os
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pdftotext
from gensim import corpora
from nltk.corpus import stopwords
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

from data.preprocess import read_pdf, preprocess_doc
from main import setup_logger
from datasets import load_dataset


def get_stop_words():
    with open('stopwords.txt') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines += ['fig', 'may', 'using', 'approach', 'need']
    return set(lines).union(STOPWORDS).union(stopwords.words('english'))


def generate_wordcloud(sentences):
    s1 = [' '.join(s) for s in sentences]
    s2 = ' '.join(s1)
    stop_words = get_stop_words()
    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(s2)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png')


def topic_analysis(sentences):
    stopwords = get_stop_words()
    sentences_wo_stopwords = [[t for t in sentence if t not in stopwords] for sentence in sentences]
    id2word = corpora.Dictionary(sentences_wo_stopwords)
    corpus = [id2word.doc2bow(text) for text in sentences_wo_stopwords]
    # number of topics
    num_topics = 20  # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)  # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())


def number_of_pages(file):
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f, raw=True)
    return len(pdf)

def modelling_statistics(args):
    logger = logging.getLogger()
    files = glob.glob(args.corpus + "/**/*.pdf", recursive=True)
    logger.info(f'Number of pdfs: {len(files)}')

    sentences = []
    pdfs_parsed = 0
    pages = 0
    tokens_per_venue = defaultdict(int)
    for f in tqdm(files, desc='Preprocessing files'):
        try:
            content = read_pdf(f)
        except:
            logging.getLogger().info(f'Error in file {f}')
            continue
        pdfs_parsed += 1
        pages += number_of_pages(f)
        tokens = preprocess_doc(content)
        sentences += tokens

        venue = f.split('/')[3]
        tokens_per_venue[venue] += sum([len(s) for s in tokens])

    # generate_wordcloud(sentences)
    # topic_analysis(sentences)
    unique_tokens = set([])
    for sentence in sentences:
        for token in sentence:
            unique_tokens.add(token)

    logger.info(f'Number of sentences: {len(sentences)}')
    sent_lens = [len(sentence) for sentence in sentences]
    number_tokens = sum(sent_lens)
    logger.info(f'Number of tokens: {number_tokens}')
    logger.info(f'Number of unique tokens: {len(unique_tokens)}')
    logger.info(f'Sentence length avg+-std: {np.mean(sent_lens):.4f} +- {np.std(sent_lens):.4f}')
    logger.info(f'Number of pdfs parsed: {pdfs_parsed}')
    logger.info(f'Number of pages: {pages}')
    logger.info(tokens_per_venue)

    with open('tokens_per_venue.pkl', 'wb') as f:
        pickle.dump(tokens_per_venue, f)

def so_statistics(args):
    logger = logging.getLogger()
    with open("./stackoverflow.txt", 'r') as file:
        lines = [l.strip() for l in file.readlines()]
        # Directorio donde estan todas las categorias
        directories = [d for d in os.listdir('/data2/sodump/all') if
                       os.path.isdir(os.path.join('/data2/sodump/all', d))]
        # Quiero su interseccion
        common_elements = set(lines) & set(directories)
        # Por cada directorio deseado me quedo con Posts y Comments
        dataset = []
        events_ = ('start', 'end')  # Yield on the start and end of a tag
        events_ns = ('start', 'start-ns', 'end', 'end-ns')  # Yield on start and end of tags and namespaces
        campos_body = []
        campos_text = []
        if "Posts" in element:
            # with open(postpath, 'r') as posts:
            # Create an an iterable

            for event, e in ET.iterparse(postpath, events=events_):
                if e.tag == "posts":
                    continue
                if event == 'start':
                    body = e.get("Body")
                    re.sub(clean, '', body)
                    # print(body)
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
                    # print(text)
                    if text is None:
                        raise ValueError(str(e))
                    if random.random() < prob:
                        campos_text.append(text)
                if event == 'end':
                    e.clear()

        # campos_body = random.sample(campos_body, round(len(campos_body) * (args.sample_size / 100)))
        # campos_text = random.sample(campos_text, round(len(campos_text) * (args.sample_size / 100)))
        dataset += campos_body + campos_text

    tokenized_files = []
    cnt = 0
    for content in dataset:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
        tokens = preprocess_doc(content)

        tokenized_files += tokens
    # generate_wordcloud(sentences)
    # topic_analysis(sentences)
    unique_tokens = set([])
    for sentence in tokenized_files:
        for token in sentence:
            unique_tokens.add(token)

    logger.info(f'Number of topics: {len(common_elements)}')
    logger.info(f'Number of posts/comments: {len(dataset)}')
    logger.info(f'Number of sentences: {len(tokenized_files)}')
    sent_lens = [len(sentence) for sentence in tokenized_files]
    number_tokens = sum(sent_lens)
    logger.info(f'Number of tokens: {number_tokens}')
    logger.info(f'Number of unique tokens: {len(unique_tokens)}')
    logger.info(f'Sentence length avg+-std: {np.mean(sent_lens):.4f} +- {np.std(sent_lens):.4f}')

def se_statistics(args):
    logger = logging.getLogger()
    with open("./selection_technical.txt", 'r') as file:
        lines = [l.strip() for l in file.readlines()]
        # Directorio donde estan todas las categorias
        directories = [d for d in os.listdir('/data2/sodump/all') if
                       os.path.isdir(os.path.join('/data2/sodump/all', d))]
        # Quiero su interseccion
        common_elements = set(lines) & set(directories)
        # Por cada directorio deseado me quedo con Posts y Comments
        dataset = []
        events_ = ('start', 'end')  # Yield on the start and end of a tag
        events_ns = ('start', 'start-ns', 'end', 'end-ns')  # Yield on start and end of tags and namespaces
        campos_body = []
        campos_text = []
        if "Posts" in element or True:
            # with open(postpath, 'r') as posts:
            # Create an an iterable

            for event, e in ET.iterparse(postpath, events=events_):
                if e.tag == "posts":
                    continue
                if event == 'start':
                    body = e.get("Body")
                    re.sub(clean, '', body)
                    # print(body)
                    if body is None:
                        raise ValueError(str(e))
                    if random.random() < prob:
                        campos_body.append(body)
                if event == 'end':
                    e.clear()
        elif "Comments" in element or True:
            for event, e in ET.iterparse(commentpath, events=events_):
                if e.tag == "comments":
                    continue
                if event == 'start':
                    text = e.get("Text")
                    re.sub(clean, '', text)
                    # print(text)
                    if text is None:
                        raise ValueError(str(e))
                    if random.random() < prob:
                        campos_text.append(text)
                if event == 'end':
                    e.clear()

        # campos_body = random.sample(campos_body, round(len(campos_body) * (args.sample_size / 100)))
        # campos_text = random.sample(campos_text, round(len(campos_text) * (args.sample_size / 100)))
        dataset += campos_body + campos_text

    tokenized_files = []
    cnt = 0
    for content in dataset:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt)
        tokens = preprocess_doc(content)

        tokenized_files += tokens
    # generate_wordcloud(sentences)
    # topic_analysis(sentences)
    unique_tokens = set([])
    for sentence in tokenized_files:
        for token in sentence:
            unique_tokens.add(token)

    logger.info(f'Number of topics: {len(common_elements)}')
    logger.info(f'Number of posts/comments: {len(dataset)}')
    logger.info(f'Number of sentences: {len(tokenized_files)}')
    sent_lens = [len(sentence) for sentence in tokenized_files]
    number_tokens = sum(sent_lens)
    logger.info(f'Number of tokens: {number_tokens}')
    logger.info(f'Number of unique tokens: {len(unique_tokens)}')
    logger.info(f'Sentence length avg+-std: {np.mean(sent_lens):.4f} +- {np.std(sent_lens):.4f}')


def wiki_statistics(args):
    logger = logging.getLogger()
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    logger.info(f'Number of pdfs: {len(wikipedia_dataset)}')

    tokenized_files = []
    for split_name, split_dataset in tqdm(wikipedia_dataset.items(), desc='Preprocessing wiki splits'):
        for row in tqdm(split_dataset, desc='Preprocessing wiki files'):
            tokenized_files += preprocess_doc(row["text"][:int(len(row["text"]) * 0.15)])
    unique_tokens = set([])
    for sentence in tokenized_files:
        for token in sentence:
            unique_tokens.add(token)

    logger.info(f'Number of topics: {len(common_elements)}')
    logger.info(f'Number of posts/comments: {len(dataset)}')
    logger.info(f'Number of sentences: {len(tokenized_files)}')
    sent_lens = [len(sentence) for sentence in tokenized_files]
    number_tokens = sum(sent_lens)
    logger.info(f'Number of tokens: {number_tokens}')
    logger.info(f'Number of unique tokens: {len(unique_tokens)}')
    logger.info(f'Sentence length avg+-std: {np.mean(sent_lens):.4f} +- {np.std(sent_lens):.4f}')

def main(args):
    modelling_statistics(args)
    #wiki_statistics(args)
    se_statistics(args)
    #so_statistics(args)

if __name__ == '__main__':
    parser = ArgumentParser(description='Script for exploring the corpus')
    parser.add_argument('--corpus', default='./docs/modelling',
                        help='Path to the w2v dataset')
    parser.add_argument('--log_file', default='info_exploring.log',
                        help='Log file')
    args = parser.parse_args()
    setup_logger(args.log_file)
    main(args)
