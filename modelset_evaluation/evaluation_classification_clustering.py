import itertools
import logging
import os
from collections import defaultdict
from re import finditer

import numpy as np
import torch
from modelset import load
from scikit_posthocs import posthoc_wilcoxon
from scipy import stats
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, v_measure_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
import random

from w2v.w2v import load_model, MODELS

MODELSET_HOME = os.path.join(os.path.expanduser('~'), 'modelset')
logger = logging.getLogger()


def get_multiset(data):
    multiset = defaultdict(int)
    for w in data:
        multiset[w.lower()] += 1
    return multiset


def filter_categories(labels, ids, thresh=10):
    # calculate cats with less than 10 elements
    cats, counts = np.unique(labels, return_counts=True)

    ignore = []
    for j, cat in enumerate(cats):
        if counts[j] < thresh:
            ignore.append(cat)

    # ignore cats with few elements
    final_good_ids = []
    for i, label in enumerate(labels):
        if label not in ignore:
            final_good_ids.append(ids[i])

    return final_good_ids


def set_up_modelset(args):
    # load dataset and generate dataframe
    dataset = load(modeltype=args.model_type, selected_analysis=['stats'])
    modelset_df = dataset.to_normalized_df(min_occurrences_per_category=args.min_occurrences_per_category)
    if args.remove_duplicates:
        ids = list(modelset_df['id'])
        corpus = [dataset.as_txt(i) for i in ids]
        corpus_multiset = [get_multiset(tokenizer(doc)) for doc in corpus]
        representatives = list(get_duplicates(corpus_multiset, ids, args.t0, args.t1).keys())
        modelset_df_no_duplicates = modelset_df[modelset_df['id'].isin(representatives)]
        ids = list(modelset_df_no_duplicates['id'])
        labels = list(modelset_df_no_duplicates['category'])
        ids = filter_categories(labels, ids, thresh=args.min_occurrences_per_category)
        modelset_df_no_duplicates_good_cat = modelset_df_no_duplicates[modelset_df_no_duplicates['id'].isin(ids)]
        return modelset_df_no_duplicates_good_cat, dataset
    return modelset_df, dataset


def jaccard_keys(multi1, multi2):
    x = multi1.keys()
    y = multi2.keys()
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)


def jaccard_generalized(multi1, multi2):
    sum_num = 0
    sum_den = 0
    for k in multi1:
        if k in multi2:
            sum_num += min(multi1[k], multi2[k])
            sum_den += max(multi1[k], multi2[k])
        else:
            sum_den += multi1[k]
    for k in multi2:
        if k not in multi1:
            sum_den += multi2[k]
    return float(sum_num) / float(sum_den)


def get_duplicates(multisets, ids, t0, t1):
    dup = {}
    words = {}
    for j, id in tqdm(enumerate(ids), desc='Duplicates main loop'):
        words[id] = multisets[j]
        bdup = False
        for id2 in dup:
            for id3 in dup[id2] + [id2]:
                if (jaccard_keys(words[id], words[id3]) > t0 and
                        jaccard_generalized(words[id], words[id3]) > t1):
                    dup[id2].append(id)
                    bdup = True
                    break
            if bdup:
                break
        if not bdup:
            dup[id] = []
    return dup


def tokenizer(doc):
    def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    words = doc.split()
    words = [w2 for w1 in words for w2 in w1.split('_') if w2 != '']
    words = [w2.lower() for w1 in words for w2 in camel_case_split(w1) if w2 != '']
    return words

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_features_w2v(doc, model, dim=300):
    words = [w for w in tokenizer(doc) if w in model.key_to_index]
    if len(words) == 0:
        logger.info(f'All zeros in a meta-model')
        return np.zeros(dim)
    vectors = np.stack([model[w] for w in words])
    return np.mean(vectors, axis=0)


def get_features_fasttext(doc, model, dim=300):
    # words = [w for w in tokenizer(doc) if w in model]
    words = [w for w in tokenizer(doc)]
    if len(words) == 0:
        logger.info(f'All zeros in a meta-model')
        return np.zeros(dim)
    vectors = np.stack([model[w] for w in words])
    return np.mean(vectors, axis=0)

def get_features_fasttext_bin(doc, model, dim=300):
    # words = [w for w in tokenizer(doc) if w in model]
    words = [w for w in tokenizer(doc)]
    if len(words) == 0:
        logger.info(f'All zeros in a meta-model')
        return np.zeros(dim)
    vectors = np.stack([model.wv[w] for w in words])
    return np.mean(vectors, axis=0)

def get_features_roberta(doc, model_tok, dim=300):
    model, tokenizer_hf = model_tok
    model.train()
    words = [w for w in tokenizer(doc)]
    words = ' '.join(words)

    inputs = tokenizer_hf([words], max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = torch.mean(model(**inputs).last_hidden_state[0], dim=0).detach().numpy()
        assert embeddings.shape == (768, )
    return embeddings


def get_vocab_modelset(corpus):
    return set([w for doc in corpus for w in tokenizer(doc)])


CS = [0.01, 0.1, 1, 10, 100]
KERNELS = ["rbf", "linear"]
COMB_HYPERPARAMETERS = list(itertools.product(CS, KERNELS))


def best_hyperparams(results):
    best_k = None
    best_v = -1
    for k, v in results.items():
        if np.mean(v) > best_v:
            best_k = k
            best_v = np.mean(v)
    return best_k


def evaluation_metamodel_classification(args):
    modelset_df, dataset = set_up_modelset(args)
    ids = list(modelset_df['id'])
    labels = list(modelset_df['category'])

    # get features and categories
    logger.info(f'Number of models {len(modelset_df)}')
    logger.info(f'Number of categories {len(np.unique(labels))}')
    corpus = [dataset.as_txt(i) for i in ids]
    X_models = {}
    for m in MODELS:
        print(m)
        seed_everything(args.seed)
        w2v_model = load_model(m, args.embeddings_out)
        if m == 'roberta':
            X_models[m] = np.array([get_features_roberta(doc, w2v_model, args.dim_embed) for doc in corpus])
        #elif m == "fasttext-mde":
        #    X_models[m] = np.array([get_features_fasttext(doc, w2v_model, args.dim_embed) for doc in corpus])
        elif "fasttext" in m or m == 'stackoverflow_modeling' or m == 'fasttext_wikipedia_modelling':
            X_models[m] = np.array([get_features_fasttext_bin(doc, w2v_model, args.dim_embed) for doc in corpus])
        else:
            X_models[m] = np.array([get_features_w2v(doc, w2v_model, args.dim_embed) for doc in corpus])
        # X_models[m] = X_models[m] / np.linalg.norm(X_models[m], axis=1, ord=2)[:, np.newaxis]

    # kfold
    skf = StratifiedKFold(n_splits=args.folds, random_state=args.seed, shuffle=True)
    scores = defaultdict(lambda: defaultdict(list))
    for train_index, test_index in tqdm(skf.split(corpus, labels),
                                        desc='Iteration over folds', total=args.folds):
        for m in MODELS:
            seed_everything(args.seed)
            X = X_models[m]
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = np.array(labels)[train_index], np.array(labels)[test_index]

            for c, kernel in COMB_HYPERPARAMETERS:
                model = SVC(random_state=args.seed, C=c, kernel=kernel)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                balanced_acc = balanced_accuracy_score(y_val, y_pred)
                scores[m][f'{c}_{kernel}'].append(balanced_acc)

    logger.info('------Best hyperparameters------')
    for m in MODELS:
        logger.info(f'B. Accuracy for {m}: {best_hyperparams(scores[m])}')

    scores = {x: scores[x][best_hyperparams(scores[x])] for x in MODELS}
    results = {x: np.mean(y) for x, y in scores.items()}

    logger.info('------Results------')
    for m in MODELS:
        logger.info(f'B. Accuracy for {m}: {results[m]}')
    logger.info('------Tests with adjustment------')
    logger.info(stats.friedmanchisquare(*[scores[m] for m in MODELS] ))
    p_adjust = 'bonferroni'
    logger.info(f'\n{posthoc_wilcoxon([scores[m] for m in MODELS], p_adjust=p_adjust)}')

    logger.info('------Tests without adjustment------')
    logger.info(f'\n{posthoc_wilcoxon([scores[m] for m in MODELS], p_adjust=None)}')


def evaluation_metamodel_clustering(args):
    modelset_df, dataset = set_up_modelset(args)
    ids = list(modelset_df['id'])
    labels = list(modelset_df['category'])

    # get features and categories
    logger.info(f'Number of models {len(modelset_df)}')
    logger.info(f'Number of categories {len(np.unique(labels))}')
    corpus = [dataset.as_txt(i) for i in ids]
    X_models = {}
    for m in MODELS:
        seed_everything(args.seed)
        w2v_model = load_model(m, args.embeddings_out)
        print(m)
        if m == 'roberta':
            X_models[m] = np.array([get_features_roberta(doc, w2v_model, args.dim_embed) for doc in corpus])
        #elif m == "fasttext-mde":
        #    X_models[m] = np.array([get_features_fasttext(doc, w2v_model, args.dim_embed) for doc in corpus])
        elif "fasttext" in m or m == 'stackoverflow_modeling' or m == 'fasttext_wikipedia_modelling':
            X_models[m] = np.array([get_features_fasttext_bin(doc, w2v_model, args.dim_embed) for doc in corpus])
        else:
            X_models[m] = np.array([get_features_w2v(doc, w2v_model, args.dim_embed) for doc in corpus])
    results = defaultdict(list)
    for m in tqdm(MODELS, desc='Iteration over word embeddings'):
        for i in range(1, 11):
            model = KMeans(random_state=args.seed + i, verbose=False, n_clusters=len(np.unique(labels)))
            model.fit(X_models[m])
            y_pred = model.labels_
            results[m].append(v_measure_score(labels_true=np.array(labels), labels_pred=y_pred))

    logger.info('------Results------')
    for m in MODELS:
        logger.info(f'V-measure for {m}: {np.mean(results[m])}')
    logger.info('------Tests------')
    logger.info(stats.friedmanchisquare(*[results[m] for m in MODELS]))
    p_adjust = 'bonferroni'
    logger.info(f'\n{posthoc_wilcoxon([results[m] for m in MODELS], p_adjust=p_adjust)}')

    logger.info('------Tests without adjustment------')
    logger.info(f'\n{posthoc_wilcoxon([results[m] for m in MODELS], p_adjust=None)}')
