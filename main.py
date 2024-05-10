import logging
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from prettytable import PrettyTable

from data.preprocess import preprocess_dataset, preprocess_dataset_metamodel_concepts, preprocess_sodump, preprocess_wikipedia
from modelset_evaluation.evaluation_classification_clustering import evaluation_metamodel_classification, \
    evaluation_metamodel_clustering
from modelset_evaluation.evaluation_metamodel_concepts import evaluation_concepts, example_recommendation
from w2v.w2v import training_word2vec, training_fasttext, test_similarity_word2vec, MODELS
import fasttext

def main(args):
    logger = logging.getLogger()
    if args.train:
        logger.info('Start preprocessing')
        tokenized_files = preprocess_dataset(args)
        logger.info(f'Finish preprocessing, number of lines: {len(tokenized_files)}')
        if args.modelType == 'word':
            training_word2vec(args, tokenized_files)
        elif args.modelType == 'fasttext':
            training_fasttext(args, tokenized_files)
    if args.train_sodump or args.train_sedump:
        logger.info('Start preprocessing')
        if args.train_sodump:
            tags = args.tags
            selection_file = args.so_selection
        else:
            tags = None
            selection_file = args.se_selection
            
        tokenized_files = preprocess_sodump(args, selection_file, tags)
        logger.info(f'Finish preprocessing, number of lines: {len(tokenized_files)}')
        if args.modelType == 'word':
            training_word2vec(args, tokenized_files)
        elif args.modelType == 'fasttext':
            training_fasttext(args, tokenized_files)
    if args.train_all:
        logger.info('Start preprocessing')
        tokenized_files1 = preprocess_sedump(args, args.se_selection, None)                
        tokenized_files2 = preprocess_sodump(args, args.so_selection, args.tags)
        tokenized_files3 = preprocess_dataset(args)

        tokenized_all = tokenized_files1 + tokenized_files2 + tokenized_files3
        logger.info(f'Finish preprocessing, number of lines: {len(tokenized_all)}')        
        random.shuffle(tokenized_all)
        if args.modelType == 'word':
            training_word2vec(args, tokenized_all)
        elif args.modelType == 'fasttext':
            training_fasttext(args, tokenized_all)
    if args.train_modeling_wiki:
        logger.info('Start preprocessing')
        tokenized_files = preprocess_wikipedia(args)
        tokenized_files2 = []
        #tokenized_files2 = preprocess_dataset(args)
        logger.info(f'Finish preprocessing, number of lines: {len(tokenized_files)}')
        tokenized_all = tokenized_files + tokenized_files2
        random.shuffle(tokenized_all)
        if args.modelType == 'word':
            training_word2vec(args, tokenized_all)
        elif args.modelType == 'fasttext':
            training_fasttext(args, tokenized_all)
    if args.test_similarity:
        test_similarity_word2vec(args)
    if args.evaluation_metamodel_classification:
        evaluation_metamodel_classification(args)
    if args.evaluation_metamodel_clustering:
        evaluation_metamodel_clustering(args)
    if args.evaluation_metamodel_concepts:
        items = preprocess_dataset_metamodel_concepts(args)
        logger.info(f'Finish preprocessing, number of items: {len(items)}')
        logger.info(f'Context type: {args.context_type}')
        avg = np.mean([len(item['recommendations']) for item in items])
        std = np.std([len(item['recommendations']) for item in items])
        logger.info(f'Avg recommendations: {avg}+-{std}')
        evaluation_concepts(args, items)
    if args.example_recommendation:
        example_recommendation(args)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def setup_logger(log_file):
    # std out
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # file out
    if log_file is not None:
        file = logging.FileHandler(log_file)
        file.setLevel(level=logging.INFO)
        formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
        file.setFormatter(formatter)
        logger.addHandler(file)


def print_command_and_args():
    logger = logging.getLogger()
    logger.info('COMMAND: {}'.format(' '.join(sys.argv)))
    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info('Configuration:\n{}'.format(config_table))

if __name__ == '__main__':
    parser = ArgumentParser(description='Script for w2v w2v')
    parser.add_argument('--training_dataset', default='./docs',
                        help='Path to the w2v dataset')
    parser.add_argument('--training_dataset_concepts', default='./java/parser/out',
                        help='Path to the concepts modelset dataset')
    parser.add_argument('--modelType', default='fasttext',
                        help='Model to train the tokenizer. word or fasttext')
    parser.add_argument('--embeddings_out', default='./out',
                        help='root folder of the embeddings')
    parser.add_argument('--folder_out_embeddings', default='skip_gram_modelling',
                        help='Path to the where the embeddings are/will be saved')
    parser.add_argument('--seed', help='seed.', type=int, default=123)  # 12390
    parser.add_argument('--folds', help='folds.', type=int, default=10)
    parser.add_argument('--model', default='word2vec-mde',
                        help='w2v model',
                        choices=MODELS)
    parser.add_argument('--w2v_algorithm', default='skip_gram',
                        help='w2v model to train embeddings',
                        choices=['skip_gram', 'cbow'])
    parser.add_argument('--model_type', default='ecore',
                        help='ecore or uml',
                        choices=['ecore', 'uml'])
    parser.add_argument('--dim_embed', type=int, default=300,
                        help='Set the size of the embeddings')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Set the sample percentage')
    parser.add_argument('--train', help='Train w2v', action='store_true')
    parser.add_argument('--train_sodump', help='Train with the StackOverflow dump', action='store_true')
    parser.add_argument('--train_sedump', help='Train with the StackOverflow dump', action='store_true')    
    parser.add_argument('--so_selection', help='Selection of stackexchange items', default='./selection_technical.txt')
    parser.add_argument('--se_selection', help='Selection of stackoverflow items', default='./selection_stackoverflow.txt')
    parser.add_argument('--tags', help='StackOverflow/Exchange tags', default=None)
    parser.add_argument('--train_all', help='Train w2v', action='store_true')
    parser.add_argument('--train_modeling_wiki', help='Train w2v', action='store_true')
    parser.add_argument('--test_similarity', help='Test similarity w2v', action='store_true')
    parser.add_argument('--evaluation_metamodel_classification', help='Evaluate embeddings in metamodel classification',
                        action='store_true')
    parser.add_argument('--evaluation_metamodel_clustering', help='Evaluate embeddings in metamodel clustering',
                        action='store_true')
    parser.add_argument('--evaluation_metamodel_concepts', help='Evaluate embeddings in metamodel concept '
                                                                'recommendation',
                        action='store_true')
    parser.add_argument('--context_type', help='Recommendation of structural features, classifiers, or literals',
                        default='EClass', choices=['EClass', 'EPackage', 'EEnum'])
    parser.add_argument('--example_recommendation', help='Example recommendation',
                        action='store_true')
    parser.add_argument('--remove_duplicates', help='Remove duplicate models', action='store_true')
    parser.add_argument('--min_occurrences_per_category', help='Min occurences per category.', type=int, default=10)
    parser.add_argument("--t0", dest="t0", help="t0 threshold.", type=float, default=0.8)
    parser.add_argument("--t1", dest="t1", help="t1 threshold.", type=float, default=0.7)
    parser.add_argument('--device', default='cpu',
                        help='cpu or cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--log_file', default='info.log',
                        help='Log file')
    args = parser.parse_args()

    seed_everything(args.seed)
    setup_logger(args.log_file)
    print_command_and_args()

    main(args)
