import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scikit_posthocs import posthoc_wilcoxon
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from w2v.w2v import load_model, MODELS

KEYS_CONTEXT_TYPE = {"EPackage": 0, "EClass": 1, "EEnum": 2}
logger = logging.getLogger()


class RecommenderModel(nn.Module):
    def __init__(self, vectors, device):
        super().__init__()
        self.in_layer = nn.Embedding.from_pretrained(torch.from_numpy(vectors).float())
        self.in_layer.weight.requires_grad = False
        self.embedding_weights = torch.from_numpy(vectors).float().to(device)
        self.embedding_weights.requires_grad = False
        self.linear_layers_in = nn.Parameter(
            data=torch.zeros(vectors.shape[1], 128))  # vectors.shape[1]
        # nn.init.uniform_(self.linear_layers_in, -0.05, 0.05)
        nn.init.xavier_normal_(self.linear_layers_in)
        self.linear_layers_out = nn.Parameter(
            data=torch.zeros(vectors.shape[1], 128))  # vectors.shape[1]
        # nn.init.uniform_(self.linear_layers_out, -0.05, 0.05)
        nn.init.xavier_normal_(self.linear_layers_out)
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


THRESH = [3, 5, 10]


def evaluation_concepts(args, items):
    # load all models
    models = []
    for m in MODELS:
        w2v_model = load_model(m, args.embeddings_out)
        models.append(w2v_model)

    results_recalls = {}
    for model_name_id, w2v_model in enumerate(models):
        keyed_items = [items_to_keys(item, w2v_model) for item in items]

        train, test = train_test_split(keyed_items, test_size=0.3, random_state=args.seed)
        train_data_loader = DataLoader(dataset=train,
                                       batch_size=32,
                                       shuffle=True,
                                       collate_fn=collate_fn,
                                       num_workers=0)
        recommender_model = RecommenderModel(np.array(w2v_model.vectors), args).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, recommender_model.parameters()), lr=0.001)
        criterion = nn.BCELoss(reduction='none')

        # training phase
        recommender_model.train()
        for epoch in range(1, 6):
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

                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, recommender_model.parameters()),
                                         max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                training_loss += loss.item()
            training_loss = training_loss / len(train_data_loader)
            logger.info(f'[epoch {epoch}] train loss: {round(training_loss, 4)}')

        logger.info(f'Saving model checkpoint {MODELS[model_name_id]}')

        os.makedirs("./models", exist_ok=True)
        duplication = 'not_duplicated' if args.remove_duplicates else 'duplicated'
        torch.save(recommender_model.state_dict(),
                   f'models/{MODELS[model_name_id]}_{args.context_type}_{duplication}.bin')

        # evaluation phase
        recommender_model.eval()
        test_data_loader = DataLoader(dataset=test,
                                      batch_size=32,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      num_workers=0)
        recalls = defaultdict(list)
        for step, batch in enumerate(tqdm(test_data_loader,
                                          desc='[testing batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            _, context, recommendations_indices = batch
            # output_lsfm: b x V
            output_lsfm = recommender_model(context.to(args.device))
            top10 = torch.topk(output_lsfm, k=10, dim=1).indices.cpu().detach().tolist()
            recommendations_indices = recommendations_indices.tolist()
            recommendations_indices = [[r1 for r1 in r if r1 != -1] for r in recommendations_indices]
            for thr in THRESH:
                for pred, true in zip(top10, recommendations_indices):
                    intersection = [v for v in pred[0:thr] if v in true]
                    recalls[thr].append(float(len(intersection)) / len(true))
        for thr in THRESH:
            logger.info(f'[Evaluation] recall@{thr}: {round(np.mean(recalls[thr]), 4)}')
        results_recalls[MODELS[model_name_id]] = recalls

    logger.info('------Tests------')
    for thr in THRESH:
        logger.info(f'Recall@{thr}')
        logger.info(stats.friedmanchisquare(*[results_recalls[m][thr] for m in MODELS]))
        p_adjust = 'bonferroni'
        logger.info(f'\n{posthoc_wilcoxon([results_recalls[m][thr] for m in MODELS], p_adjust=p_adjust)}')


def example_recommendation(args):
    w2v_model = load_model(args.model, args.embeddings_out)
    recommender_model = RecommenderModel(np.array(w2v_model.vectors), args).to(args.device)
    duplication = 'not_duplicated' if args.remove_duplicates else 'duplicated'
    recommender_model.load_state_dict(torch.load(f'models/{args.model}_{args.context_type}_{duplication}.bin'))
    recommender_model.eval()

    logger.info(f'Introduce as input your {args.context_type} name')
    logger.info(f'To exit press ctrl + d')
    for line in sys.stdin:
        word = line.rstrip()
        if word not in w2v_model.key_to_index:
            logger.info(f'Word {word} not in vocab')
            logger.info(f'Introduce as input your {args.context_type}')
            continue
        context = torch.tensor([w2v_model.key_to_index[word]])
        output_lsfm = recommender_model(context.to(args.device))
        top10 = torch.topk(output_lsfm, k=10, dim=1).indices.cpu().detach().tolist()[0]
        logger.info(f'-------Recommendations for {word}-------')
        for r in top10:
            logger.info(f'{w2v_model.index_to_key[r]}')
        logger.info(f'Introduce as input your {args.context_type}')
