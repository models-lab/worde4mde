import logging
import os

from gensim.models import Word2Vec, KeyedVectors

logger = logging.getLogger()


def training_w2v(args, sentences):
    model = Word2Vec(sentences=sentences, vector_size=300,
                     min_count=10, window=10, workers=10, epochs=10,
                     seed=args.seed)
    logger.info(f'Vocab size: f{len(model.wv.key_to_index)}')

    if not os.path.exists(args.embeddings_out):
        os.makedirs(args.embeddings_out)
    model.wv.save(os.path.join(args.embeddings_out, 'vectors.kv'))
    # print(list(model.wv.key_to_index))
    # print(model.wv['state'])


def test_word2vec(args):
    reloaded_word_vectors = KeyedVectors.load(os.path.join(args.embeddings_out, 'vectors.kv'))
    for word in ['state', 'sql',
                 'assignment', 'petri',
                 'father', 'name', 'atl',
                 'graph', 'classroom']:
        logger.info(f'Most similar {word}: {reloaded_word_vectors.most_similar(positive=[word])}')

