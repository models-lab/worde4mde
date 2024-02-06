import os
import shutil
import tarfile

import gdown
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

CACHE_PATH = os.path.join(os.path.expanduser('~'), '.worde4mde')
EMBEDDINGS_FOLDER = 'embeddings'
URL = 'https://drive.google.com/uc?id=13wkOuD1Is5rzDxRzCcfDd42rrzaMtowB'
SGRAM_MDE_PATH = os.path.join('out', 'skip_gram_modelling', 'skip_gram_vectors.kv')
GLOVE_MDE_PATH = os.path.join('out', 'glove_modelling', 'vectors.txt')


def clear_cache():
    if os.path.exists(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


def download_embeddings():
    os.makedirs(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER), exist_ok=True)
    tar_gz_file = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'embeddings.tar.gz')
    gdown.download(URL, tar_gz_file, quiet=False)
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(path=os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER))


def load_embeddings(embedding_model='sgram-mde'):
    if not os.path.exists(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER)):
        download_embeddings()
    if embedding_model == 'sgram-mde':
        reloaded_word_vectors = KeyedVectors.load(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, SGRAM_MDE_PATH))
        return reloaded_word_vectors
    elif embedding_model == 'glove-mde':
        glove_file = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, GLOVE_MDE_PATH)
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
        return reloaded_word_vectors
    else:
        raise ValueError(f'Unknown embedding model: {embedding_model}')
