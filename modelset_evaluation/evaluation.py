import logging
import os
from collections import defaultdict
from re import finditer

import numpy as np
from modelset import load
from scikit_posthocs import posthoc_wilcoxon
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

from w2v.w2v import load_model, MODELS

MODELSET_HOME = os.path.join(os.path.expanduser('~'), 'modelset')
logger = logging.getLogger()


def set_up_modelset():
    # load dataset and generate dataframe
    dataset = load(modeltype='ecore', selected_analysis=['stats'])
    modelset_df = dataset.to_normalized_df()
    return modelset_df, dataset


def tokenizer(doc):
    def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    words = doc.split('\n')
    words = [w2 for w1 in words for w2 in w1.split('_') if w2 != '']
    words = [w2.lower() for w1 in words for w2 in camel_case_split(w1) if w2 != '']
    return words


def get_features_w2v(doc, model, dim=300):
    words = [w for w in tokenizer(doc) if w in model.key_to_index]
    if len(words) == 0:
        logger.info('All zeros in a meta-model')
        return np.zeros(dim)
    vectors = np.stack([model[w] for w in words])
    return np.mean(vectors, axis=0)


def evaluation_metamodel_classification(args):
    modelset_df, dataset = set_up_modelset()

    # get features and categories
    ids = list(modelset_df['id'])
    labels = list(modelset_df['category'])
    corpus = [dataset.as_txt(i) for i in ids]
    X_models = {}
    for m in MODELS:
        if m == 'word2vec-mde':
            w2v_model = load_model(m, args.embeddings_out)
        else:
            w2v_model = load_model(m)
        X_models[m] = np.array([get_features_w2v(doc, w2v_model) for doc in corpus])
        # X_models[m] = X_models[m] / np.linalg.norm(X_models[m], axis=1, ord=2)[:, np.newaxis]

    # kfold
    skf = StratifiedKFold(n_splits=args.folds, random_state=args.seed, shuffle=True)
    scores = defaultdict(list)
    for train_index, test_index in tqdm(skf.split(corpus, labels),
                                        desc='Iteration over folds', total=args.folds):
        for m in MODELS:
            X = X_models[m]
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = np.array(labels)[train_index], np.array(labels)[test_index]

            model = SVC(random_state=args.seed, C=20, kernel='linear')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            scores[m].append(balanced_acc)

    results = {x: np.mean(y) for x, y in scores.items()}

    logger.info('------Results------')
    for m in MODELS:
        logger.info(f'B. Accuracy for {m}: {results[m]}')
    logger.info('------Tests------')
    logger.info(stats.friedmanchisquare(*[scores[m] for m in MODELS]))
    logger.info(posthoc_wilcoxon([scores[m] for m in MODELS],
                                 p_adjust='bonferroni'))
