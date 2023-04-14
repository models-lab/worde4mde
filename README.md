# WordE4MDE

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

## Download trained embeddings 🚀

To download the WordE4MDE embeddings just run the following:

```shell
./scripts/download_embeddings.sh
```

## Exploring embeddings 📋

Let us consider the following list of words:
```python
['state', 'atl', 'dsl', 'grammar',
'petri', 'statechart', 'ecore', 'epsilon',
'qvt', 'transformation']
```

The commands below compute, for each word model, the top 10 similar words 
for each word of the previous list:

```shell
python main.py --test_similarity --model glove-mde
python main.py --test_similarity --model skip_gram-mde
python main.py --test_similarity --model glove-wiki-gigaword-300
python main.py --test_similarity --model word2vec-google-news-300
```

## Using the embeddings for meta-model classification, clustering and recommendation 📋

Meta-model classification task:
```shell
python main.py --evaluation_metamodel_classification --remove_duplicates
```

Meta-model clustering task:
```shell
python main.py --evaluation_metamodel_clustering --remove_duplicates
```

Meta-model concepts task (the parser is applied to the ModelSet dataset, 
and then the recommendation systems are trained and evaluated):
```shell
cd java/parser
mvn compile
mvn exec:java
cd ../..
python main.py --evaluation_metamodel_concepts --remove_duplicates --device cpu --context_type EEnum
python main.py --evaluation_metamodel_concepts --remove_duplicates --device cpu --context_type EPackage
python main.py --evaluation_metamodel_concepts --remove_duplicates --device cpu --context_type EClass
```

Example of recommendations:
```shell
python main.py --example_recommendation --model glove-mde --context_type {EClass, EPackage, EEnum} --remove_duplicates
python main.py --example_recommendation --model skip_gram-mde --context_type {EClass, EPackage, EEnum} --remove_duplicates
python main.py --example_recommendation --model glove-wiki-gigaword-300 --context_type {EClass, EPackage, EEnum} --remove_duplicates
python main.py --example_recommendation --model word2vec-google-news-300 --context_type {EClass, EPackage, EEnum} --remove_duplicates
```
