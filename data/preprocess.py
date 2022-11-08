import glob
import re

import pdftotext
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm


def read_pdf(file):
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f)
    all_pdf = "\n".join(pdf)
    return all_pdf


def preprocess_sentence(string):
    # lower case
    string = string.lower()

    # remove \n
    string = re.sub(r'\n', ' ', string).strip()

    # remove numbers
    string = re.sub(r'\d+', '', string)

    # remove punctuation
    string = re.sub(r'[^\w\s]', '', string).strip()

    # tokenization
    return word_tokenize(string)


def preprocess_doc(string):
    # remove \n
    string = re.sub(r'\n', ' ', string).strip()

    # split in sentences
    sentences = [preprocess_sentence(s) for s in sent_tokenize(string)]

    # remove short sentences
    return [s for s in sentences if len(s) > 5]


def preprocess_dataset(args):
    files = glob.glob(args.training_dataset + "/**/*.pdf", recursive=True)
    result = []
    for f in tqdm(files, desc='Preprocessing files'):
        content = read_pdf(f)
        tokens = preprocess_doc(content)
        result += tokens
    return result
