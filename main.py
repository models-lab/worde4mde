import logging
import random
from argparse import ArgumentParser

import numpy as np

from data.preprocess import preprocess_dataset
from w2v.w2v import training_word2vec, test_similarity_word2vec, test_kmeans_word2vec, visualize_embeddings


def main(args):
    logger = logging.getLogger()
    if args.train:
        logger.info('Start preprocessing')
        tokenized_files = preprocess_dataset(args)
        logger.info(f'Finish preprocessing, number of lines: {len(tokenized_files)}')
        training_word2vec(args, tokenized_files)
    if args.test_similarity:
        test_similarity_word2vec(args)
    if args.test_kmeans:
        test_kmeans_word2vec(args)
    if args.visualize_embeddings:
        visualize_embeddings(args)


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
    parser.add_argument('--model', default='word2vec-mde',
                        help='Path to the where the embeddings will be saved',
                        choices=['word2vec-mde',
                                 'glove-wiki-gigaword-300',
                                 'word2vec-google-news-300'])
    parser.add_argument('--train', help='Train w2v', action='store_true')
    parser.add_argument('--test_similarity', help='Test similarity w2v', action='store_true')
    parser.add_argument('--test_kmeans', help='Test kmeans w2v', action='store_true')
    parser.add_argument('--visualize_embeddings', help='Tsne', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    setup_logger()

    main(args)
