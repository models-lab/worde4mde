import logging
import os

import gensim.downloader as api
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import save_facebook_model, load_facebook_model
import fasttext
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger()

DEFAULT_VECTORS_NAME = 'vectors.kv'
GLOVE_VECTORS_NAME = 'vectors.txt'
SKIP_GRAM_VECTORS = 'skip_gram_vectors.kv'
CBOW_VECTORS = 'cbow_vectors.kv'
MODELS = [
    'glove-wiki-gigaword-300',
    # 'glove-mde',
    # 'word2vec-google-news-300',
    'skip_gram-mde',
    # 'so_word2vec',
    # sgram-se',
    # 'average',
    # 'average_sgramglove',
    # 'roberta',
    # 'sgram-se',
    # 'fasttext_bin',
    # 'sgram-se-mde',
    # 'fasttext50',
    # 'fasttext100',
    # 'fasttext200',
    # 'fasttext_bin',
    # 'fasttext400',
    # 'fasttext500',
    # 'sgram-mde-50',
    # 'sgram-mde-100',
    # 'sgram-mde-200',
    # 'skip_gram-mde',
    # 'sgram-mde-400',
    # 'sgram-mde-500',
    # 'stackoverflow_modeling',
    # 'fasttext_wikipedia_modelling',
    # 'fasttext-se',
    #
]

PATHS = {
    'skip_gram-mde': 'out/skip_gram_modelling/skip_gram_vectors.kv',
    'cbow-mde': 'out/cbow_modelling/cbow_vectors.kv',
    'glove-mde': 'out/glove_modelling/vectors.txt',
    'fasttext-mde': 'embeddings/fasttext-mde/skip_gram_vectors.kv',
    'so_word2vec': 'embeddings/so_word2vec/SO_vectors_200.bin',
    'average': 'embeddings/average_gloves/average_gloves.txt',
    'average_sgramglove': 'embeddings/average_sgramglove/average_gloves.txt',
    'sgram-sodump': 'out/skip_gram_sodump/skip_gram_vectors.kv',
    'fasttext_bin': 'out/fasttext_bin/fasttext_model.bin',
    'roberta': 'bert-modeling/checkpoint-61140',
    'sgram-se-mde': 'out/skip_gram_all/skip_gram_vectors.kv',
    'fasttext-se-mde': 'out/sodump_all_modelling/fasttext_model.bin',
    'fasttext50': 'out/fasttext_modeling_50/fasttext_model.bin',
    'fasttext100': 'out/fasttext_modeling_100/fasttext_model.bin',
    'fasttext200': 'out/fasttext_modeling_200/fasttext_model.bin',
    'fasttext400': 'out/fasttext_modeling_400/fasttext_model.bin',
    'fasttext500': 'out/fasttext_modeling_500/fasttext_model.bin',
    'sgram-mde-50': 'out/word_modeling_50/skip_gram_vectors.kv',
    'sgram-mde-100': 'out/word_modeling_100/skip_gram_vectors.kv',
    'sgram-mde-200': 'out/word_modeling_200/skip_gram_vectors.kv',
    'sgram-mde-400': 'out/word_modeling_400/skip_gram_vectors.kv',
    'sgram-mde-500': 'out/word_modeling_500/skip_gram_vectors.kv',
    'stackoverflow_modeling': '/data/worde4mde/embeddings/fasttext_stackoverflow_modelling/fasttext_model.bin',
    'fasttext_wikipedia_modelling': 'out/fasttext_wikipedia_modelling/fasttext_model.bin',
    'fasttext-se': 'out/fasttext-se/fasttext_model.bin',
    'sgram-se': 'out/sgram-se/skip_gram_vectors.kv',
}


def training_word2vec(args, sentences):
    if args.w2v_algorithm == 'skip_gram':
        sg = 1
        vectors_name = SKIP_GRAM_VECTORS
    else:
        sg = 0
        vectors_name = CBOW_VECTORS

    model = Word2Vec(sentences=sentences, vector_size=args.dim_embed,
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

    model = FastText(sentences=sentences, vector_size=args.dim_embed,
                     min_count=10, window=10, workers=10, epochs=20,
                     seed=args.seed, sg=sg)
    logger.info(f'Vocab size: f{len(model.wv.key_to_index)}')

    output_folder = os.path.join(args.embeddings_out, args.folder_out_embeddings)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # model.wv.save(os.path.join(output_folder, vectors_name))
    save_facebook_model(model, os.path.join(output_folder, 'fasttext_model.bin'))


def load_model(model, embeddings_out=None):
    print(model)
    #    'fasttext-sodump',
    #    'fasttext-all',

    if model == 'skip_gram-mde' or model == 'sgram-se':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'cbow-mde':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'glove-mde':
        glove_file = os.path.join(PATHS[model])
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    elif model == 'fasttext-mde':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'so_word2vec':
        reloaded_word_vectors = KeyedVectors.load_word2vec_format(PATHS[model], binary=True)
    elif model == 'average':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'average_sgramglove':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'sgram-sodump':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif 'fasttext' in model:
        reloaded_word_vectors = load_facebook_model(PATHS[model])
        # reloaded_word_vectors = fasttext.load_model(PATHS[model])
        # reloaded_word_vectors = KeyedVectors.load_word2vec_format(PATHS[model], binary=True)
    elif model == 'sgram-all':
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'roberta':
        model = AutoModel.from_pretrained(PATHS[model], output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        reloaded_word_vectors = (model, tokenizer)
    elif "sgram-mde" in model:
        reloaded_word_vectors = KeyedVectors.load(PATHS[model])
    elif model == 'stackoverflow_modeling' or model == 'fasttext_wikipedia_modelling':
        reloaded_word_vectors = load_facebook_model(PATHS[model])
    else:
        reloaded_word_vectors = api.load(model)
    return reloaded_word_vectors


def test_similarity_word2vec(args):
    reloaded_word_vectors = load_model(args.model, args.embeddings_out)
    dict = {}
    colorDict = {}
    bag = ['petrinet', 'node', 'place', 'transition', 'arc', 'ptarc', 'tparc', 'token', 'weight',
           'source', 'target', 'nodes', 'arcs']
    if len(bag) != 0:
        vectors = []
        for x in bag:
            vectors.append(reloaded_word_vectors[x])
        words = bag.copy()
        vectors = np.array(list(vectors))
    else:
        for i, word in enumerate(bag):
            if word in reloaded_word_vectors.key_to_index:
                m_similar = ', '.join([p[0] for p in reloaded_word_vectors.most_similar(positive=[word])])
                for p in reloaded_word_vectors.most_similar(positive=[word]):
                    dict[p[0]] = reloaded_word_vectors[p[0]]
                    colorDict[p[0]] = i
                logger.info(f'Top 10 similar words to \"{word}\": {m_similar}')
            else:
                logger.info(f'Word {word} not in vocab')

        words = list(dict.keys())
        vectors = np.array(list(dict.values()))

    pca = PCA(n_components=2, random_state=123)  # Adjust the number of components as needed
    vectors_2d = pca.fit_transform(vectors)

    #tsne = TSNE(n_components=2, random_state=123, perplexity=1e-4)
    #tsne = TSNE(n_components=2, random_state=20, init='pca', perplexity=1e-4)
    print(vectors)
    print(len(vectors))
    #vectors_2d = tsne.fit_transform(vectors)
    # color_values = [colorDict[word] for word in words]
    # Define a colormap
    # cmap = cm.get_cmap('tab10')  # You can choose any other colormap from Matplotlib

    # Normalize color values to be between 0 and 1
    # norm = plt.Normalize(min(color_values), max(color_values))

    # Get colors from the colormap
    # colors = [cmap(norm(value)) for value in color_values]

    plt.figure(figsize=(10, 8))
    # plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='.', c=colors)
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='.')
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=8)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Word Vectors')
    plt.savefig("embeddings_tsne.png")
