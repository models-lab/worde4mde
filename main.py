import logging
import random
from argparse import ArgumentParser

import numpy as np

from data.preprocess import preprocess_dataset
from w2v.w2v import training_w2v, test_word2vec


def main(args):
    logger = logging.getLogger()
    if args.train:
        logger.info('Start preprocessing')
        tokenized_files = preprocess_dataset(args)
        logger.info('Finish preprocessing')
        training_w2v(args, tokenized_files)
    if args.test:
        test_word2vec(args)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for w2v w2v')
    parser.add_argument('--training_dataset', default='./docs',
                        help='Path to the w2v dataset')
    parser.add_argument('--embeddings_out', default='./out',
                        help='Path to the where the embeddings will be saved')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    parser.add_argument('--train', help='If w2v phase', action='store_true')
    parser.add_argument('--test', help='If w2v phase', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    setup_logger()

    main(args)
