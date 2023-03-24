from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

from django import forms

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors

import os


# class ModelLoader:

def load_model(model, embeddings_file=None):
    if model == 'word2vec-mde':
        reloaded_word_vectors = KeyedVectors.load(embeddings_file)
    else:
        reloaded_word_vectors = api.load(model)
    return reloaded_word_vectors

VECTORS_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'vectors')

SKIPGRAM_MODEL = load_model('word2vec-mde',
                            os.path.join(VECTORS_FOLDER, 'skip_gram_modelling', 'skip_gram_vectors.kv'))
GLOVE_MDE_MODEL = None
# GLOVE_MDE_MODEL = load_model('word2vec-mde',
#                             os.path.join(os.path.dirname(__file__), '..', '..', 'vectors', 'glove_modelling',
#                                          'vectors.txt'))
GLOVE_MODEL = None  # load_model('glove-wiki-gigaword-300')
WORD2VEC_MODEL = None  # load_model('word2vec-google-news-300')


class SearchForm(forms.Form):
    term = forms.CharField(label='Term', max_length=128)

    SKIPGRAM_MDE = 'skip_gram_mde'
    GLOVE_MDE = 'glove_mde'
    GLOVE = 'glove'
    WORD2VEC = 'word2vec'
    CHOICES = (
        (SKIPGRAM_MDE, u"Skip-gram MDE"),
        (GLOVE_MDE, u"GloVe MDE"),
        (GLOVE, u"GloVe"),
        (WORD2VEC, u"Word2Vec"),
    )
    selected_model = forms.ChoiceField(choices=CHOICES)

    def get_model(self):
        model = self.cleaned_data['selected_model']
        if model == self.SKIPGRAM_MDE:
            return SKIPGRAM_MODEL
        elif model == self.GLOVE_MDE:
            return GLOVE_MDE_MODEL
        elif model == self.WORD2VEC:
            return WORD2VEC_MODEL
        else:
            return GLOVE_MODEL


class RecommendationForm(SearchForm):
    ECLASS = 'EClass'
    EENUM = 'EEnum'
    EPACKAGE = 'EPackage'
    CONTEXTS = (
        (ECLASS, u"EClass"),
        (EENUM, u"EEnum"),
        (EPACKAGE, "EPackage")
    )
    contexts = forms.ChoiceField(choices=CONTEXTS)


def index(request):
    context = {
        'similar_active': 'active',
        'recommendation_active': None,
    }

    form = RecommendationForm()
    context['form_rec'] = form

    form = SearchForm()
    context['form_sim'] = form
    return render(request, 'app/index.html', context)


def similar_words(request):
    context = {
        'similar_words': [],
        'similar_active': 'active',
        'recommendation_active': None,
    }
    form = RecommendationForm()
    context['form_rec'] = form

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SearchForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            term = form.cleaned_data['term'].lower()
            model = form.get_model()

            if term in model.key_to_index:
                similar = model.most_similar(positive=[term])
                for word, score in similar:
                    context['similar_words'].append(word)
            else:
                context['similar_words'].append('Word not in vocabulary')

            context['form_sim'] = form
            return render(request, 'app/index.html', context)
    else:
        form = SearchForm()
        context['form_sim'] = form
        return render(request, 'app/index.html', context)


# add folder to python path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modelset_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation_metamodel_concepts import RecommenderModel
import torch
import numpy as np


def load_recommendation_model(model, loaded_model, context_type):
    # w2v_model = load_model(model, embeddings_file)
    recommender_model = RecommenderModel(np.array(loaded_model.vectors), "cpu").to("cpu")
    rec_file = os.path.join(VECTORS_FOLDER, 'recommendation', f'{model}_{context_type}_not_duplicated.bin')
    recommender_model.load_state_dict(torch.load(rec_file))
    recommender_model.eval()
    return recommender_model


REC_SKIPGRAM_ECLASS = load_recommendation_model('skip_gram-mde', SKIPGRAM_MODEL, 'EClass')
REC_SKIPGRAM_EPACKAGE = load_recommendation_model('skip_gram-mde', SKIPGRAM_MODEL, 'EPackage')
REC_SKIPGRAM_EENUM = load_recommendation_model('skip_gram-mde', SKIPGRAM_MODEL, 'EEnum')


def get_recommendation_model(model, context_type):
    if model == 'skip_gram_mde':
        if context_type == 'EClass':
            return REC_SKIPGRAM_ECLASS
        elif context_type == 'EPackage':
            return REC_SKIPGRAM_EPACKAGE
        elif context_type == 'EEnum':
            return REC_SKIPGRAM_EENUM

    print("Not loaded")
    return None


def recommend(rec_model, w2v_model, context_name):
    context = torch.tensor([w2v_model.key_to_index[context_name]])
    output_lsfm = rec_model(context.to("cpu"))
    top10 = torch.topk(output_lsfm, k=10, dim=1).indices.cpu().detach().tolist()[0]
    return [w2v_model.index_to_key[r] for r in top10]


def recommendation(request):
    context = {
        'recommendations': [],
        'similar_active': None,
        'recommendation_active': 'active',
    }
    form = SearchForm()
    context['form_sim'] = form

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        form = RecommendationForm(request.POST)
        if form.is_valid():
            term = form.cleaned_data['term'].lower()
            selected_model = form.cleaned_data['selected_model']
            context_type = form.cleaned_data['contexts']
            rec_model = get_recommendation_model(selected_model, context_type)
            w2v_model = form.get_model()

            if term not in w2v_model.key_to_index:
                context['recommendations'].append('Word not in vocabulary')
            else:
                context['recommendations'] = recommend(rec_model, w2v_model, term)

            context['form_rec'] = form
            return render(request, 'app/index.html', context)
    else:
        form = RecommendationForm()
        context['form_rec'] = form
        return render(request, 'app/index.html', context)
