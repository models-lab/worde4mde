import glob
import logging
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from data.preprocess import read_pdf, preprocess_doc
from main import setup_logger


def main(args):
    logger = logging.getLogger()

    files = glob.glob(args.corpus + "/**/*.pdf", recursive=True)
    logger.info(f'Number of pdfs: {len(files)}')

    sentences = []
    for f in tqdm(files, desc='Preprocessing files'):
        content = read_pdf(f)
        tokens = preprocess_doc(content)
        sentences += tokens

    logger.info(f'Number of sentences: {len(sentences)}')
    sent_lens = [len(sentence) for sentence in sentences]
    number_tokens = sum(sent_lens)
    logger.info(f'Number of tokens: {number_tokens}')
    logger.info(f'Sentence length avg+-std: {np.mean(sent_lens):.4f} +- {np.std(sent_lens):.4f}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for exploring the corpus')
    parser.add_argument('--corpus', default='./docs',
                        help='Path to the w2v dataset')
    parser.add_argument('--log_file', default='info_exploring.log',
                        help='Log file')
    args = parser.parse_args()
    setup_logger(args.log_file)
    main(args)
