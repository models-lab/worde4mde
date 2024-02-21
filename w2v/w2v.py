import logging
import os

import gensim.downloader as api
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

logger = logging.getLogger()

DEFAULT_VECTORS_NAME = 'vectors.kv'
GLOVE_VECTORS_NAME = 'vectors.txt'
SKIP_GRAM_VECTORS = 'skip_gram_vectors.kv'
CBOW_VECTORS = 'cbow_vectors.kv'
MODELS = [
    'glove-wiki-gigaword-300',
    'skip_gram-mde',
    'glove-mde',
    'word2vec-google-news-300',
    'fasttext',
    'so_word2vec'
        ]

PATHS = {
    'skip_gram-mde': 'out/skip_gram_modelling/skip_gram_vectors.kv',
    'cbow-mde': 'out/cbow_modelling/cbow_vectors.kv',
    'glove-mde': 'out/glove_modelling/vectors.txt',
    'fasttext': 'embeddings/fasttext/skip_gram_vectors.kv',
    'so_word2vec': 'embeddings/so_word2vec/SO_vectors_200.bin'
}


def training_word2vec(args, sentences):
    if args.w2v_algorithm == 'skip_gram':
        sg = 1
        vectors_name = SKIP_GRAM_VECTORS
    else:
        sg = 0
        vectors_name = CBOW_VECTORS

    model = Word2Vec(sentences=sentences, vector_size=300,
                     min_count=10, window=10, workers=10, epochs=20,
                     seed=args.seed, compute_loss=True, sg=sg)
    logger.info(f'Vocab size: f{len(model.wv.key_to_index)}')

    output_folder = os.path.join(args.embeddings_out, args.folder_out_embeddings)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model.wv.save(os.path.join(output_folder, vectors_name))
    # print(list(model.wv.key_to_index))
    # print(model.wv['state'])

def training_fasttext(args, sentences):
    if args.w2v_algorithm == 'skip_gram':
        sg = 1
        vectors_name = SKIP_GRAM_VECTORS
    else:
        sg = 0
        vectors_name = CBOW_VECTORS

    model = FastText(sentences=sentences, vector_size=300,
                     min_count=10, window=10, workers=10, epochs=20,
                     seed=args.seed, sg=sg)
    logger.info(f'Vocab size: f{len(model.wv.key_to_index)}')

    output_folder = os.path.join(args.embeddings_out, args.folder_out_embeddings)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model.wv.save(os.path.join(output_folder, vectors_name))


def load_model(model, embeddings_out=None):
    if model == 'skip_gram-mde':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'cbow-mde':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'glove-mde':
        glove_file = os.path.join(PATHS[model])
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    elif model == 'fasttext':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'so_word2vec':
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(PATHS[model], binary=True)
    elif model == 'average':
        reloaded_word_vectors = "a"
    else:
        reloaded_word_vectors = api.load(model)
    return reloaded_word_vectors


def test_similarity_word2vec(args):
    reloaded_word_vectors = load_model(args.model, args.embeddings_out)
    for word in ['state', 'atl', 'dsl', 'grammar',
                 'petri', 'statechart', 'ecore', 'epsilon',
                 'qvt', 'transformation']:
        if word in reloaded_word_vectors.key_to_index:
            m_similar = ', '.join([p[0] for p in reloaded_word_vectors.most_similar(positive=[word])])
            logger.info(f'Top 10 similar words to \"{word}\": {m_similar}')
        else:
            logger.info(f'Word {word} not in vocab')
