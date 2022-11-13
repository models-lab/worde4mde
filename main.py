import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

from data.preprocess import preprocess_dataset, preprocess_dataset_metamodel_concepts
from modelset_evaluation.evaluation_classification_clustering import evaluation_metamodel_classification, \
    evaluation_metamodel_clustering
from modelset_evaluation.evaluation_metamodel_concepts import evaluation_concepts
from w2v.w2v import training_word2vec, test_similarity_word2vec, \
    test_kmeans_word2vec, visualize_embeddings, MODELS


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
    if args.evaluation_metamodel_classification:
        evaluation_metamodel_classification(args)
    if args.evaluation_metamodel_clustering:
        evaluation_metamodel_clustering(args)
    if args.evaluation_metamodel_concepts:
        items = preprocess_dataset_metamodel_concepts(args)
        logger.info(f'Finish preprocessing, number of items: {len(items)}')
        avg = np.mean([len(item['recommendations']) for item in items])
        std = np.std([len(item['recommendations']) for item in items])
        logger.info(f'Avg recommendations: {avg}+-{std}')
        evaluation_concepts(args, items)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


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
    parser.add_argument('--training_dataset_concepts', default='./java/parser/out',
                        help='Path to the concepts modelset dataset')
    parser.add_argument('--embeddings_out', default='./out',
                        help='Path to the where the embeddings will be saved')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    parser.add_argument('--folds', help='folds.', type=int, default=10)
    parser.add_argument('--model', default='word2vec-mde',
                        help='Path to the where the embeddings will be saved',
                        choices=MODELS)
    parser.add_argument('--model_type', default='ecore',
                        help='ecore or uml',
                        choices=['ecore', 'uml'])
    parser.add_argument('--train', help='Train w2v', action='store_true')
    parser.add_argument('--test_similarity', help='Test similarity w2v', action='store_true')
    parser.add_argument('--test_kmeans', help='Test kmeans w2v', action='store_true')
    parser.add_argument('--visualize_embeddings', help='Tsne', action='store_true')
    parser.add_argument('--evaluation_metamodel_classification', help='Evaluate embeddings in metamodel classification',
                        action='store_true')
    parser.add_argument('--evaluation_metamodel_clustering', help='Evaluate embeddings in metamodel clustering',
                        action='store_true')
    parser.add_argument('--evaluation_metamodel_concepts', help='Evaluate embeddings in metamodel concept '
                                                                'recommendation',
                        action='store_true')
    parser.add_argument('--remove_duplicates', help='Remove duplicate models', action='store_true')
    parser.add_argument('--min_occurrences_per_category', help='Min occurences per category.', type=int, default=10)
    parser.add_argument("--t0", dest="t0", help="t0 threshold.", type=float, default=0.8)
    parser.add_argument("--t1", dest="t1", help="t1 threshold.", type=float, default=0.7)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.seed)
    setup_logger()

    main(args)
