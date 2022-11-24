import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from w2v.w2v import load_model, MODELS

KEYS_CONTEXT_TYPE = {"EPackage": 0, "EClass": 1, "EEnum": 2}
logger = logging.getLogger()


class RecommenderModel(nn.Module):
    def __init__(self, vectors, args):
        super().__init__()
        self.in_layer = nn.Embedding.from_pretrained(torch.from_numpy(vectors).float())
        self.in_layer.weight.requires_grad = False
        self.embedding_weights = torch.from_numpy(vectors).float().to(args.device)
        self.embedding_weights.requires_grad = False
        self.linear_layers_in = nn.Parameter(
            data=torch.zeros(vectors.shape[1], vectors.shape[1]))
        nn.init.uniform_(self.linear_layers_in, -0.05, 0.05)
        self.linear_layers_out = nn.Parameter(
            data=torch.zeros(vectors.shape[1], vectors.shape[1]))
        nn.init.uniform_(self.linear_layers_out, -0.05, 0.05)
        self.sfm = nn.Softmax(dim=1)

    def forward(self, context):
        context_emb = self.in_layer(context)
        context_transform = torch.matmul(context_emb, self.linear_layers_in)
        r_transformation = torch.matmul(self.embedding_weights, self.linear_layers_out)
        return self.sfm(torch.matmul(context_transform, torch.transpose(r_transformation, 0, 1)))


def collate_fn(batch):
    context = [item["context"] for item in batch]
    context_type = [item["context_type"] for item in batch]
    recommendations_indices = [item["recommendations_indices"][0:len(batch)] for item in batch]
    recommendations_indices = [r + ([-1] * (len(batch) - len(r))) for r in recommendations_indices]
    return torch.tensor(context_type), torch.tensor(context), torch.tensor(recommendations_indices)


def batch_indices_to_zeros(indices, model):
    list_zeros = []
    for b in indices:
        zeros = [0] * len(model.key_to_index)
        for i in b:
            if i != -1:
                zeros[i] = 1
        list_zeros.append(zeros)
    return torch.tensor(list_zeros)


def items_to_keys(item, model):
    indices = []
    for r in item["recommendations"]:
        indices.append(model.key_to_index[r])
    item_new = {"context": model.key_to_index[item["context"]],
                "recommendations_indices": indices,
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

    for model_name_id, w2v_model in enumerate(models):
        keyed_items = [items_to_keys(item, w2v_model) for item in items]

        train, test = train_test_split(keyed_items, test_size=0.3)
        train_data_loader = DataLoader(dataset=train,
                                       batch_size=32,
                                       shuffle=True,
                                       collate_fn=collate_fn,
                                       num_workers=0)
        recommender_model = RecommenderModel(np.array(w2v_model.vectors), args).to(args.device)
        optimizer = torch.optim.Adam(recommender_model.parameters(), lr=0.001)
        criterion = nn.BCELoss(reduction='none')

        # training phase
        recommender_model.train()
        for epoch in range(1, 4):
            training_loss = 0.
            for step, batch in enumerate(tqdm(train_data_loader,
                                              desc='[training batch]',
                                              bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
                _, context, recommendations_indices = batch
                recommendations = batch_indices_to_zeros(recommendations_indices, w2v_model)
                # output_lsfm: b x V
                output_lsfm = recommender_model(context.to(args.device))
                loss = criterion(output_lsfm, recommendations.float().to(args.device))
                loss = loss.sum(dim=1)
                loss = loss.mean(dim=0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_loss += loss.item()
            training_loss = training_loss / len(train_data_loader)
            logger.info(f'[epoch {epoch}] train loss: {round(training_loss, 4)}')

        logger.info(f'Saving model checkpoint {MODELS[model_name_id]}')
        torch.save(recommender_model.state_dict(), f'{MODELS[model_name_id]}.bin')

        # evaluation phase
        recommender_model.eval()
        test_data_loader = DataLoader(dataset=test,
                                      batch_size=32,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      num_workers=0)
        recalls = []
        for step, batch in enumerate(tqdm(test_data_loader,
                                          desc='[testing batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            _, context, recommendations_indices = batch
            # output_lsfm: b x V
            output_lsfm = recommender_model(context.to(args.device))
            top10 = torch.topk(output_lsfm, k=10, dim=1).indices.cpu().detach().tolist()
            recommendations_indices = recommendations_indices.tolist()
            recommendations_indices = [[r1 for r1 in r if r1 != -1] for r in recommendations_indices]
            for pred, true in zip(top10, recommendations_indices):
                intersection = [v for v in pred if v in true]
                recalls.append(float(len(intersection)) / len(true))
        logger.info(f'[Evaluation] recall: {round(np.mean(recalls), 4)}')
