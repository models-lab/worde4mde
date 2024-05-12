import itertools
import logging
import math
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from mlxtend.evaluate import mcnemar_table, mcnemar
from nltk.corpus import stopwords
from sklearn.metrics import cohen_kappa_score

from main import setup_logger
from modelset_evaluation.evaluation_classification_clustering import set_up_modelset, tokenizer
from w2v.w2v import load_model, MODELS


def get_vocab_modelset(corpus):
    vocab_modelset = defaultdict(int)
    for doc in corpus:
        for w in tokenizer(doc):
            vocab_modelset[w] += 1
    return vocab_modelset  # set([w for doc in corpus for w in tokenizer(doc)])


def get_vocab_per_model(corpus):
    vocabs = []
    for doc in corpus:
        vocabs.append(set(tokenizer(doc)))
    return vocabs


def cohen_h(p1, p2):
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def main(args):
    logger = logging.getLogger()

    modelset_df, dataset = set_up_modelset(args)
    ids = list(modelset_df['id'])
    corpus = [dataset.as_txt(i) for i in ids]
    vocab_modelset = get_vocab_modelset(corpus)
    vocabs_per_model = get_vocab_per_model(corpus)
    logger.info(f'Number of models {len(modelset_df)}')
    logger.info(f'Vocab modelset size {len(vocab_modelset)}')

    logger.info('Coverage and vocab sizes')
    ones_intersection = {}
    proportions = {}
    intersections = {}
    for m in MODELS:
        model = load_model(m)
        intersection = [w for w in vocab_modelset if w in model.key_to_index]
        ones_intersection[m] = [1 if w in model.key_to_index else 0 for w in vocab_modelset]
        logger.info(f'For model {m} vocab size: {len(model.key_to_index)}')
        logger.info(f'For model {m} coverage: {float(len(intersection)) / (float(len(vocab_modelset))):.4f}')
        proportions[m] = float(len(intersection)) / (float(len(vocab_modelset)))
        intersections[m] = intersection

        coverage_per_model = []
        for vocab in vocabs_per_model:
            intersection = [w for w in vocab if w in model.key_to_index]
            coverage_per_model.append(float(len(intersection)) / (float(len(vocab))))
        logger.info(f'For model {m} model granularity coverage: {np.mean(coverage_per_model):.4f}')
        logger.info(f'Vocab sizes: {np.mean([len(v) for v in vocabs_per_model]):.4f}')

    logger.info('Cohen h and Kappa')
    for m1, m2 in itertools.combinations(MODELS, 2):
        o1 = np.array(ones_intersection[m1])
        o2 = np.array(ones_intersection[m2])
        truth = np.array([1 for _ in range(len(o1))])
        tb = mcnemar_table(y_target=truth,
                           y_model1=o1,
                           y_model2=o2)
        chi2, p = mcnemar(ary=tb, corrected=True)
        logger.info(f'{m1} vs {m2}: {p}')
        # logger.info(f'{tb}')

        p1 = proportions[m1]
        p2 = proportions[m2]
        logger.info(f'Cohen h: {cohen_h(p1, p2)}')
        logger.info(f'Kappa: {cohen_kappa_score(o1, o2)}')

    logger.info('Intersections and word examples')
    for m1 in MODELS:
        for m2 in MODELS:
            if m1 != m2:
                diff = set(intersections[m1]) - set(intersections[m2])
                logger.info(f'{m1} - {m2}: {float(len(diff)) / float(len(vocab_modelset)):.4f}')
                diff = [w for w in diff if w.isalpha() and w not in set(stopwords.words('english'))]
                diff.sort(key=lambda x: -vocab_modelset[x])
                logger.info(f'Words in diff: {[(w, vocab_modelset[w]) for w in diff][0:10]}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for exploring the embeddings')
    #parser.add_argument('--corpus', default='./docs/modelling',
    #                    help='Path to the w2v dataset')
    parser.add_argument('--log_file', default='info_exploring_vocab.log',
                        help='Log file')
    parser.add_argument('--remove_duplicates', help='Remove duplicate models', action='store_true')
    parser.add_argument('--min_occurrences_per_category', help='Min occurences per category.', type=int, default=10)
    parser.add_argument("--t0", dest="t0", help="t0 threshold.", type=float, default=0.8)
    parser.add_argument("--t1", dest="t1", help="t1 threshold.", type=float, default=0.7)
    parser.add_argument('--model_type', default='ecore',
                        help='ecore or uml',
                        choices=['ecore', 'uml'])

    args = parser.parse_args()
    setup_logger(args.log_file)
    main(args)
