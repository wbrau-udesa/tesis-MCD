from importlib import reload  # Python 3.4+
import gc
import os 
import pickle

import pandas as pd 
import numpy as np
from itertools import islice
from itertools import combinations
import scipy.sparse as sp

from random import seed
from random import sample

from tqdm import tqdm
tqdm.pandas()
import time
import multiprocessing
cpus = multiprocessing.cpu_count()

import matplotlib.pyplot as plt
import seaborn as sns


import requests
import bs4
import re
import zipfile
from io import BytesIO


# Clustering 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


import kmedoids


#  Regresión y clasificación
import statsmodels.api as sm

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet, LinearRegression, QuantileRegressor, HuberRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay,f1_score, accuracy_score, precision_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import median_absolute_error, mean_squared_error

from imblearn.over_sampling import SMOTE


# NLP 
import re
import string
import unidecode

from collections import Counter 
from collections import OrderedDict

from torchtext.vocab import vocab as Vocab

from sklearn.feature_extraction.text import CountVectorizer # --- > to first obtain the Term-Document Matrix
from sklearn.feature_extraction.text import TfidfTransformer # ---> tf idf
from sklearn.pipeline import Pipeline

import spacy
from spacy.language import Language
from langdetect import detect_langs

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
#from nltk.stem import LancasterStemmer

import gensim.downloader as api

