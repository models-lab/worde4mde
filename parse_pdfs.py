import glob
from argparse import ArgumentParser

import pdftotext
from datasets import Dataset
from tqdm import tqdm


def get_pdf(f):
    with open(f, "rb") as f:
        pdf = pdftotext.PDF(f, raw=True)
        all_pdf = " ".join(pdf)
        all_pdf = all_pdf.replace("\n", " ").strip()
    return all_pdf


def main(args):
    files = glob.glob(args.pdf_folder + "/**/*.pdf", recursive=True)
    texts = []
    for f in tqdm(files):
        try:
            text = get_pdf(f)
            texts.append({"text": text, "file": f})
        except:
            print(f'Cannot parse {f}')

    dataset = Dataset.from_list(texts)
    print(dataset)
    dataset.to_json(args.output)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for getting txt files from pdfs')
    parser.add_argument('--pdf_folder')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args)
