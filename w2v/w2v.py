import logging
import os

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors

logger = logging.getLogger()

DEFAULT_VECTORS_NAME = 'vectors.kv'
MODELS = ['word2vec-mde',
          'glove-wiki-gigaword-300',
          'word2vec-google-news-300']


def training_word2vec(args, sentences):
    model = Word2Vec(sentences=sentences, vector_size=300,
                     min_count=10, window=10, workers=10, epochs=20,
                     seed=args.seed, compute_loss=True, sg=1)
    logger.info(f'Vocab size: f{len(model.wv.key_to_index)}')

    if not os.path.exists(args.embeddings_out):
        os.makedirs(args.embeddings_out)
    model.wv.save(os.path.join(args.embeddings_out, DEFAULT_VECTORS_NAME))
    # print(list(model.wv.key_to_index))
    # print(model.wv['state'])


def load_model(model, embeddings_out=None):
    if model == 'word2vec-mde':
        reloaded_word_vectors = KeyedVectors.load(os.path.join(embeddings_out, DEFAULT_VECTORS_NAME))
    else:
        reloaded_word_vectors = api.load(model)
    return reloaded_word_vectors


def test_similarity_word2vec(args):
    reloaded_word_vectors = load_model(args.model, args.embeddings_out)
    for word in ['state', 'sql', 'transition',
                 'assignment', 'petri',
                 'father', 'name', 'epsilon',
                 'graph', 'classroom', 'transformation',
                 'statechart']:
        if word in reloaded_word_vectors.key_to_index:
            logger.info(f'Most similar {word}: {reloaded_word_vectors.most_similar(positive=[word])}')
        else:
            logger.info(f'Word {word} not in vocab')
