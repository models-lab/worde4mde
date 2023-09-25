import unittest

from worde4mde import download_embeddings, clear_cache, load_embeddings


class TestWorde4mde(unittest.TestCase):
    def test_download(self):
        clear_cache()
        download_embeddings()

    def test_load_embeddings(self):
        sgram_mde = load_embeddings(embedding_model='sgram-mde')
        word = 'ecore'
        print(sgram_mde.most_similar(positive=[word]))
        print(sgram_mde[word])

        glove_mde = load_embeddings(embedding_model='glove-mde')
        word = 'ecore'
        print(glove_mde.most_similar(positive=[word]))

        print(type(glove_mde[word]))


if __name__ == '__main__':
    unittest.main()
