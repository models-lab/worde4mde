import unittest

import nltk

from data.preprocess import preprocess_sentence, read_pdf, preprocess_doc

FILE_TEST = 'docs/modelling/sosym/441118.pdf'
nltk.download('punkt')


class TestPreprocessing(unittest.TestCase):

    def test_pdftotext(self):
        print(read_pdf(FILE_TEST))

    def test_preprocess_doc_to_sent(self):
        all_pdf = read_pdf(FILE_TEST)
        print(preprocess_sentence(all_pdf))

    def test_preprocess_doc(self):
        all_pdf = read_pdf(FILE_TEST)
        print(preprocess_doc(all_pdf))


if __name__ == '__main__':
    unittest.main()
