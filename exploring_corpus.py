import glob
import logging
from argparse import ArgumentParser
from pprint import pprint

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
        pdf = pdftotext.PDF(f)
    return len(pdf)


def main(args):
    logger = logging.getLogger()

    files = glob.glob(args.corpus + "/**/*.pdf", recursive=True)
    logger.info(f'Number of pdfs: {len(files)}')

    sentences = []
    pdfs_parsed = 0
    pages = 0
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


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for exploring the corpus')
    parser.add_argument('--corpus', default='./docs/modelling',
                        help='Path to the w2v dataset')
    parser.add_argument('--log_file', default='info_exploring.log',
                        help='Log file')
    args = parser.parse_args()
    setup_logger(args.log_file)
    main(args)
