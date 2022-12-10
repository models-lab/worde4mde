# Word2vec4MDE

## Installation 🛠

### With conda (recommended)

This repo is written in Python and Java. Thus, I recommend to use conda. To initialize the conda environment,
just execute:
```shell
sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
conda env create --file=conda_venv.yml
conda activate word2vec-mde
python -m nltk.downloader all
```

After that, you need to download ModelSet dataset as all the experiments were run over this dataset.
```shell
python -m modelset.downloader
```

### Without conda

You need to install:
- Python 3.8.X
- Openjdk 1.8
- Maven 3.8.6

Generate a virtual environment and then install the requirements.

```shell
sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader all
```

After that, you need to download ModelSet dataset.
```shell
python -m modelset.downloader
```

## Train Word2vec4MDE from scratch 🚀

Download the dataset of papers and put it in a folder:
```shell
TODO
```

Run the training procedure (by default all pdfs have to be placed in a folder called `docs`):
```shell
python main.py --train
```

## Exploring embeddings 📋

Word similarity:
```shell
python main.py --test_similarity --model word2vec-mde 
```

Clustering:
```shell
python main.py --cluster_word_vectors
```

## Using the embeddings for meta-model classification, clustering and recommendation 📋

Meta-model classification task:
```shell
python main.py --evaluation_metamodel_classification
```

Meta-model clustering task:
```shell
python main.py --evaluation_metamodel_clustering
```

Meta-model concepts task:
```shell
cd java/parser
mvn compile
mvn exec:java
cd ../..
python main.py --evaluation_metamodel_concepts --device cpu
```

Example of recommendations:
```shell
python main.py --example_recommendation --model word2vec-mde
python main.py --example_recommendation --model glove-wiki-gigaword-300
python main.py --example_recommendation --model word2vec-google-news-300
```
