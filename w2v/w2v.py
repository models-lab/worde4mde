import logging
import os

import gensim.downloader as api
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer

logger = logging.getLogger()

DEFAULT_VECTORS_NAME = 'vectors.kv'
DEFAULT_FIG_NAME = 'out.pdf'
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


def training_glove(args, sentences):
    pass


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
                 'graph', 'classroom', 'transformation']:
        logger.info(f'Most similar {word}: {reloaded_word_vectors.most_similar(positive=[word])}')


def test_kmeans_word2vec(args):
    reloaded_word_vectors = load_model(args.model, args.embeddings_out)

    model = KMeans(random_state=args.seed, verbose=True)
    visualizer = KElbowVisualizer(model,
                                  metric='distortion',
                                  k=list(range(10, 110, 10)),
                                  timings=False)
    visualizer.fit(reloaded_word_vectors.vectors)
    visualizer.show(outpath=DEFAULT_FIG_NAME, clear_figure=True)


def visualize_embeddings(args):
    reloaded_word_vectors = load_model(args.model, args.embeddings_out)
    X_embedded = TSNE(n_components=2,
                      learning_rate='auto',
                      init='random',
                      perplexity=40,
                      random_state=args.seed).fit_transform(reloaded_word_vectors.vectors)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1, alpha=0.4)
    plt.savefig(DEFAULT_FIG_NAME)
