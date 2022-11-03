import unittest

import nltk
import pdftotext

from data.preprocess import preprocess_doc_to_sent, read_pdf

FILE_TEST = 'docs/441118.pdf'
nltk.download('punkt')


def get_pdf():
    with open(FILE_TEST, "rb") as f:
        pdf = pdftotext.PDF(f)
    return pdf


class TestPreprocessing(unittest.TestCase):

    def test_pdftotext(self):
        print(read_pdf(FILE_TEST))

    def test_preprocess_doc_to_sent(self):
        all_pdf = read_pdf(FILE_TEST)
        print(preprocess_doc_to_sent(all_pdf))


if __name__ == '__main__':
    unittest.main()
