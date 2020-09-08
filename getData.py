
from Bio import Entrez
import os
import requests
from os import path
import pandas as pd
import datetime
import gzip
import json
import time
import re
import sys
import urllib 
import ftplib as ftp
from dotenv import load_dotenv
import h5py
import GEOparse
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy as np
import nltk
from nltk import bigrams
import itertools
import pathlib
import difflib

# import R packages
base = importr('base')
utils = importr('GEOquery')

load_dotenv(verbose=True)

PTH = os.environ.get('PTH_A')
Entrez.email = os.environ.get('EMAIL')
API_KEY = os.environ.get('API_KEY')

#--------------------------------------------- archs4 ------------------------------------------------
'''
# load the archs4 data for human
filename = os.path.join(PTH,"human_matrix.h5")
f = h5py.File(filename,"r")  # ['data', 'info', 'meta']>
geo_accession_numbers_human = [ val.decode('utf-8') for val in f['meta']['Sample_geo_accession'] ]

# load the archs4 data for mouse
filename = os.path.join(PTH,"mouse_matrix.h5")
f = h5py.File(filename,"r")
geo_accession_numbers_mouse = [ val.decode('utf-8') for val in f['meta']['Sample_geo_accession'] ]

# combine archs4 data for human and mouse
ARCHS4_geo_accession_numbers = geo_accession_numbers_human + geo_accession_numbers_mouse
del geo_accession_numbers_mouse, geo_accession_numbers_human

# save accession numbers to file
ARCHS4_geo_accession_numbers = pd.DataFrame(ARCHS4_geo_accession_numbers)
ARCHS4_geo_accession_numbers.to_csv(os.path.join(PTH,'ARCHS4_GEO_Samples.csv'))
'''
#-----------------------------------------------------------------------------------------------------

# create a word co-occurrence matrix
def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))
    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)
    # return the matrix and the index
    return (co_occurrence_matrix, vocab_index)


# returns a list of pubmen ids that correlate with rat experiments in GEO.
def article_links(q):
  handle = Entrez.esearch(db="gds", term=q,
                              api_key=API_KEY,
                              usehistory ='y',
                              retmax = 1000000
                              )
  records = Entrez.read(handle)
  return (records['IdList'])


# collect json data from GEO
def collectData(geo_id):
    time.sleep(0.3)
    handle = Entrez.esummary(db="gds", id=geo_id, retmode="xml", api_key=API_KEY)
    records = Entrez.parse(handle)
    results = [record for record in records ]
    return(results[0])


# download files from GEO using GEOquery R package and rpy2
def downloadFiles(datasets):
    pathlib.Path(os.path.join(PTH,"samples")).mkdir(parents=True, exist_ok=True)
    for geo_dic in datasets:
        try:
            robjects.r('getGEOSuppFiles("%s", makeDirectory = TRUE, baseDir = "%s")' % (geo_dic['Accession'],os.path.join(PTH,"samples")))
        except Exception as e:
            print(e)


def clean(df_):
    df_['Title'] = df_['Title'].str.lower()
    df_['Title'] = df_['Title'].str.replace('-',' ')
    df_['Title'] = df_['Title'].str.replace('_',' ')
    df_['Title'] = df_['Title'].str.replace('+',' ')
    df_['Title'] = [word_tokenize(x) for x in df_['Title'] ]
    return(df_)
 

# check titles of samples to see if it contains control or perturbation keywords
def check_sample_title(datasets):
    appended_data = []
    control_keywords = pd.read_csv('https://raw.githubusercontent.com/MaayanLab/GEOSampleClassifier/master/data/keywords_control?token=AFKKUEP6JLSSYJ2X3UGNFXS7MEMVU', 
                                    header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()
    pert_keywords = pd.read_csv('https://raw.githubusercontent.com/MaayanLab/GEOSampleClassifier/master/data/keywords_perturbation?token=AFKKUEPHGLTMV6EXM7CUKKS7MENL6',
                                    header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()        
    for geo_dic in datasets:
        df = pd.DataFrame(geo_dic['Samples'])
        df['GEO'] = geo_dic['Accession']
        df['group'] = 'A'
        df.at[df['Title'].isin( difflib.get_close_matches(df['Title'][0], df['Title']) ),'group' ] ='B'
        df = clean(df)
        # decide if control or perturbation
        df['type'] = ""
        for i in range(len(df)):
            if any(item in pert_keywords for item in df.iloc[i]['Title']):
                df.at[i,'type'] = 'pert'
            if any(item in control_keywords for item in df.iloc[i]['Title']):
                df.at[i,'type'] = 'cntr'
        appended_data.append(df)
    return(pd.concat(appended_data))


def get_sample_landing_page(df):
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize 
    sentences = []
    for sample in df['Accession']:
        gse = GEOparse.get_GEO(geo=sample, destdir=os.path.join(PTH,'tmp'))
        text = ' '.join([" ".join(gse.metadata[key]) for key in gse.metadata.keys()])
        # tokenize
        #tokens = word_tokenize(text)
        #filtered_words = [word for word in tokens if word not in stopwords.words('english')]
        sentences = sentences + filtered_words
    generate_co_occurrence_matrix(text)


if __name__ == "__main__":
    q = '((((bulk RNA-seq) OR (bulk RNA seq)) NOT(single-cell)))'
    GEO_IDs = article_links(q)
    datasets = []
    keywords = ['rna','bulk','seq']
    blacklist = ['scrna-seq','single cell', 'single-cell']
    for i in range(0,len(GEO_IDs)):
        print(i)
        if i%50==0:
            time.sleep(2)
        try:
            geo_dic = collectData(GEO_IDs[i])
        except Exception as e:
            print(geo_dic)
            datasets.append(geo_dic)
            time.sleep(2)
        title = geo_dic['title'].lower()
        # Relevant files
        if (all(item in title for item in keywords)) and (not any(item in title for item in blacklist)):
            if geo_dic['n_samples'] >= 6 and geo_dic['n_samples'] <= 12: # limit number of samples
                print("found",i)
                datasets.append(geo_dic)

    downloadFiles(datasets)
    # for each dataset- check samples titles
    df = check_sample_title(datasets)
    get_sample_landing_page(df)
