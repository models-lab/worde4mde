import os
import shutil
import tarfile
import sys

from urllib.request import urlretrieve
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

CACHE_PATH = os.path.join(os.path.expanduser('~'), '.worde4mde')
EMBEDDINGS_FOLDER = 'embeddings'

URL = 'http://models-lab.inf.um.es/tools/worde4mde'
URL_SGRAM_MDE = URL + '/sgram-mde.tar.gz'
URL_SGRAM_MDE_SO = URL + '/sgram-mde-so.tar.gz'
URL_GLOVE_MDE = URL + '/glove-mde.tar.gz'
URL_FASTTEXT_MDE = URL + '/fasttext-mde/fasttext_model.bin'
URL_FASTTEXT_MDE_SO = URL + '/fasttext-mde-so/fasttext_model.bin'

SGRAM_MDE_PATH = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'sgram-mde', 'skip_gram_vectors.kv')
SGRAM_MDE_PATH_SO = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'sgram-mde-so', 'skip_gram_vectors.kv')
GLOVE_MDE_PATH = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'glove-mde', 'vectors.txt')
FASTTEXT_MDE_PATH = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'fasttext-mde', 'fasttext_model.bin')
FASTTEXT_MDE_PATH_SO = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'fasttext-mde-so', 'fasttext_model.bin')

def clear_cache():
    if os.path.exists(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)

def __reporthook__(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size if total_size > 0 else 0
    percent = min(100, percent)
    sys.stdout.write(f"\rDownloading: {percent:.2f}%")
    sys.stdout.flush()

def download_embeddings(url):
    os.makedirs(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER), exist_ok=True)
    tar_gz_file = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, 'embeddings.tar.gz')
    urlretrieve(url, tar_gz_file, reporthook=__reporthook__)
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(path=os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER))
    os.remove(tar_gz_file)

def download_fasttext_model(url, target_path):
    os.makedirs(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER), exist_ok=True)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urlretrieve(url, target_path, reporthook=__reporthook__)

def load_embeddings(embedding_model='sgram-mde'):
    if embedding_model == 'sgram-mde' and not os.path.exists(SGRAM_MDE_PATH):
        download_embeddings(URL_SGRAM_MDE)
    elif embedding_model == 'sgram-mde-so' and not os.path.exists(SGRAM_MDE_PATH_SO):
        download_embeddings(URL_SGRAM_MDE_SO)
    elif embedding_model == 'glove-mde' and not os.path.exists(GLOVE_MDE_PATH):
        download_embeddings(URL_GLOVE_MDE)
    elif embedding_model == 'fasttext-mde' and not os.path.exists(FASTTEXT_MDE_PATH):
        download_fasttext_model(URL_FASTTEXT_MDE, FASTTEXT_MDE_PATH)
    elif embedding_model == 'fasttext-mde-so' and not os.path.exists(FASTTEXT_MDE_PATH_SO):
        download_fasttext_model(URL_FASTTEXT_MDE_SO, FASTTEXT_MDE_PATH_SO)

    if embedding_model == 'sgram-mde':
        reloaded_word_vectors = KeyedVectors.load(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, SGRAM_MDE_PATH))
        return reloaded_word_vectors
    elif embedding_model == 'sgram-mde-so':
        reloaded_word_vectors = KeyedVectors.load(os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, SGRAM_MDE_PATH_SO))
        return reloaded_word_vectors
    elif embedding_model == 'glove-mde':
        glove_file = os.path.join(CACHE_PATH, EMBEDDINGS_FOLDER, GLOVE_MDE_PATH)
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
        return reloaded_word_vectors
    elif embedding_model == 'fasttext-mde':
        from gensim.models.fasttext import load_facebook_model
        reloaded_word_vectors = load_facebook_model(FASTTEXT_MDE_PATH)
        return reloaded_word_vectors
    elif embedding_model == 'fasttext-mde-so':
        from gensim.models.fasttext import load_facebook_model
        reloaded_word_vectors = load_facebook_model(FASTTEXT_MDE_PATH_SO)
        return reloaded_word_vectors
    else:
        raise ValueError(f'Unknown embedding model: {embedding_model}')
