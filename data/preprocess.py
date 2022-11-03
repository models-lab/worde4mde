import glob
import re

import pdftotext
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def read_pdf(file):
    with open(file, "rb") as f:
        pdf = pdftotext.PDF(f)
    all_pdf = "\n".join(pdf)
    return all_pdf


def preprocess_doc_to_sent(string):
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


def preprocess_dataset(args):
    files = glob.glob(args.training_dataset + "/*.pdf")
    result = []
    for f in tqdm(files, desc='Preprocessing files'):
        content = read_pdf(f)
        tokens = preprocess_doc_to_sent(content)
        result.append(tokens)
    return result
