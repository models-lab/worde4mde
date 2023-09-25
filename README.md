# WordE4MDE

WordE4MDE is a Python library that provides word embeddings for the MDE domain.

## Installation 🛠

Using pip:
```bash
pip install worde4mde
```

## Usage

First of all, you need to load the embeddings (currently supported: `'sgram-mde'` and `'glove-mde'`).
```python
from worde4mde import load_embeddings
sgram_mde = load_embeddings('sgram-mde')
```

The `load_embeddings` function returns a `KeyedVectors` object from 
the [gensim library](https://radimrehurek.com/gensim/models/keyedvectors.html). If you want to retrieve the most similar
words to a given word, you can use the `most_similar` function:

```python
word = 'ecore'
sgram_mde.most_similar(positive=[word])
>>> [('emf', 0.6336350440979004), ('metamodels', 0.5963817834854126), 
     ('ecorebased', 0.5922590494155884), ... 
```

Furthermore, if you want to get the embedding of a given word, just write the following:
```python
sgram_mde[word]
>>> [ 0.14674647  0.42704162  0.17717203  0.05179158  0.38020504 -0.00091264 ...
```
