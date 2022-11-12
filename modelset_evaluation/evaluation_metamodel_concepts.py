import logging

import torch
import torch.nn as nn

from w2v.w2v import load_model, MODELS

KEYS_CONTEXT_TYPE = {"EPackage": 0, "EClass": 1}
logger = logging.getLogger()


class RecommenderModel(nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.in_layers = nn.Embedding.from_pretrained(torch.from_numpy(vectors))
        self.linear_layers_in = nn.Parameter(
            data=torch.zeros(len(KEYS_CONTEXT_TYPE), vectors.shape[1], vectors.shape[1]))
        nn.init.uniform_(self.linear_layers_in, -0.05, 0.05)
        self.linear_layers_out = nn.Parameter(
            data=torch.zeros(len(KEYS_CONTEXT_TYPE), vectors.shape[1], vectors.shape[1]))
        nn.init.uniform_(self.linear_layers_out, -0.05, 0.05)

    def forward(self, x):
        pass


def items_to_keys(item, model):
    item_new = {"context": model.key_to_index[item["context"]],
                "recommendations": [model.key_to_index[r] for r in item["recommendations"]],
                "context_type": KEYS_CONTEXT_TYPE[item["context_type"]]}
    return item_new


def evaluation_concepts(args, items):
    # load all models
    models = []
    for m in MODELS:
        if m == 'word2vec-mde':
            w2v_model = load_model(m, args.embeddings_out)
        else:
            w2v_model = load_model(m)
        models.append(w2v_model)

    for w2v_model in models:
        keyed_items = [items_to_keys(item, w2v_model) for item in items]
        logger.info(keyed_items[0])
