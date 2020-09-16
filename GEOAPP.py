
from Bio import Entrez
import os
import requests
import string
from os import path
import pandas as pd
import datetime
import shutil
import gzip
import json
import time
import re
import sys
import urllib 
import ftplib as ftp
from dotenv import load_dotenv
import GEOparse
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy as np
import nltk
from nltk import bigrams
import itertools
import pathlib
import difflib
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
import networkx as nx
from nltk.stem import PorterStemmer
import gensim
from gensim.models import Word2Vec
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import connexion
from flask_cors import CORS
from flask import jsonify
nltk.download('punkt')

n = 100 #embedding size

ps = PorterStemmer()

# import R packages
base = importr('base')
utils = importr('GEOquery')

load_dotenv(verbose=True)

PTH = os.environ.get('PTH_A')
Entrez.email = os.environ.get('EMAIL')
API_KEY = os.environ.get('API_KEY')

cntrl_url = 'https://raw.githubusercontent.com/MaayanLab/GEOSampleClassifier/master/data/keywords_control?token=AFKKUENTGBLLGIOJUJZ7DUK7NIXNS'
treat_url = 'https://raw.githubusercontent.com/MaayanLab/GEOSampleClassifier/master/data/keywords_perturbation?token=AFKKUEJ5HCYYS5MKZ3ZOLHC7NIXL6'


# download files from GEO using GEOquery R package and rpy2
def downloadFiles(datasets):
    pathlib.Path(os.path.join(PTH,"samples")).mkdir(parents=True, exist_ok=True)
    for geo_dic in datasets:
        try:
            robjects.r('getGEOSuppFiles("%s", makeDirectory = TRUE, baseDir = "%s")' % (geo_dic['Accession'],os.path.join(PTH,"samples")))
        except Exception as e:
            print(e)


def clean(df_):
    df_['title'] = df_['title'].str.lower()
    df_['title'] = df_['title'].str.replace('-',' ')
    df_['title'] = df_['title'].str.replace('_',' ')
    df_['title'] = df_['title'].str.replace('+',' ')
    df_['title'] = [word_tokenize(x) for x in df_['title'] ]
    return(df_)


def set_groups(df):
    df['group'] = 0
    res = []
    for title in df['full_title']:
        title = ''.join([i for i in title if not i.isdigit()])
        res.append(title)
    df['tmp'] = res
    for i in range(0,len(df)):
        val1 = df.iloc[i]['tmp']
        x = set(difflib.get_close_matches(val1, df['tmp']))
        for t in x:
            df.at[df['tmp']==t,'group'] = i
    df['group'] = df['group'] - df['group'].min()
    del df['tmp']
    return(df)


# check titles of samples to see if it contains control or perturbation keywords
def check_sample_title(geo_dic):
    control_keywords = pd.read_csv(cntrl_url, header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()
    pert_keywords = pd.read_csv(treat_url,header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()        
    df = pd.DataFrame(geo_dic['samples'])
    df['GEO'] = geo_dic['accession']
    df['full_title'] = df['title']
    org_title = df['title'].tolist()
    df = clean(df)
    df = keep_unique_words(df)
    df = set_groups(df)
    df['full_title'] = org_title
    df['type'] = ""
    return(df)


# stemming
def stemSentence(sentence):
    del_chars = string.punctuation + '–'
    for x in del_chars:
        sentence = sentence.replace(x, ' ')
   # sentence = ''.join([i for i in sentence if not i.isdigit()]) # delete numbers
    token_words = word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(ps.stem(word.lower()))
    return (stem_sentence)


def control_or_pert(title_stem,keywords_stem):
    for w in keywords_stem:
        if w in title_stem:
            return(True)
    return(False)


# remove words that are common to all samples' titles
def keep_unique_words(df):
    GEOs = list(set(df['GEO']))
    for geo in GEOs:
        samples_title = df[df['GEO'] == geo ]['full_title'].tolist()
        samples_title = [ ' '.join(stemSentence(x)) for x in samples_title ]
        samples_title = [ set(stemSentence(x)) for x in samples_title ]
        u = set.intersection(*samples_title)
        for i in range(len(samples_title)):
            for r in u:
                samples_title[i].remove(r)
        samples_title = [' '.join(list(x)) for x in samples_title ]
        df.loc[df['GEO'] == geo ,'full_title'] = samples_title
    return(df)


# create word co-occurence graph
def create_graph(words):
    word_fd = nltk.FreqDist(words)
    bigram_fd = nltk.FreqDist(nltk.bigrams(words))
    res = [ [ x[0][0], x[0][1], x[1] ] for x in bigram_fd.most_common()]
    res = pd.DataFrame(res, columns=['frm', 'to', 'weight'])
    res = res.groupby(['frm', 'to']).agg({'weight': ['sum']})
    res.reset_index(inplace=True)
    res.columns = ['frm', 'to', 'weight']
    G=nx.from_pandas_edgelist(res, 'frm', 'to', ['weight'])
    return(G)


def serch_graph(G, keywords_stem, title_stem):
    res = []
    for k in keywords_stem:
        for w in title_stem:
            try:
                x = nx.shortest_path_length(G,w,k,weight="True")
                res.append(x)
            except:
                pass
    if len(res) == 0:
        res = 1000000
    else:
        res = min(res)
    return(res)


def process_sample_titiles(df,geo, cntrl_keywords_stem, pert_keywords_stem, G):
    for title in df[df['GEO'] == geo ]['full_title']:
        flg = False
        title_stem = stemSentence(title)
        # --- test if title contains a keyword ---------------------
        if control_or_pert(title_stem, cntrl_keywords_stem ):
            df.at[ df['full_title'] == title, 'type' ] = 'cntr'
            df.at[ df['full_title'] == title, 'dist' ] = 'keyword'
            flg = True
        if control_or_pert(title_stem, pert_keywords_stem):
            df.at[ df['full_title'] == title, 'type' ] = 'pert'
            df.at[ df['full_title'] == title, 'dist' ] = 'keyword'
            flg = True
        # --- if title dosn't contains a keyword ---------------------
        if not flg:
            cnt = serch_graph(G, cntrl_keywords_stem, title_stem)
            trt = serch_graph(G, pert_keywords_stem, title_stem)
            if  cnt < trt:
                df.at[ df['full_title'] == title, 'type' ] = 'cntr'
                df.at[ df['full_title'] == title, 'dist' ] = str(cnt)
            if cnt == trt:
                df.at[ df['full_title'] == title, 'dist' ] = str(0)
            if cnt > trt:
                df.at[ df['full_title'] == title, 'type' ] = 'pert'
                df.at[ df['full_title'] == title, 'dist' ] = str(trt)
    return(df)


def get_sample_landing_page(df):
    cntrl_keywords = pd.read_csv(cntrl_url, header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()
    cntrl_keywords_stem = stemSentence(' '.join(cntrl_keywords))
    pert_keywords = pd.read_csv(treat_url,header=None, dtype=str, keep_default_na=False)[0].str.lower().tolist()        
    pert_keywords_stem = stemSentence(' '.join(pert_keywords))
    GEOs = list(set(df['GEO']))
    org_title = df['full_title']
    df = keep_unique_words(df)
    df['dist'] = ''
    for geo in GEOs:
        gse = GEOparse.get_GEO(geo=geo, destdir=os.path.join(PTH,'tmp'))
        shutil.rmtree(os.path.join(PTH,'tmp'))
        geo_text = ' '.join([gse.metadata['title'][0], gse.metadata['summary'][0], gse.metadata['overall_design'][0]])
        df['full_title'] = df['full_title'].str.lower()
        # Create CBOW model since this is a very small curpus and low dimention
        words = stemSentence(geo_text)
        G = create_graph(words)
        df = process_sample_titiles(df,geo, cntrl_keywords_stem, pert_keywords_stem, G)
    df['title'] = org_title
    del df['full_title']
    return(df)


def fetch_geo(geo_accession):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search = base + "/esearch.fcgi?db=gds&term=%s[ACCN]&retmode=json"%geo_accession
    res = requests.get(search)
    time.sleep(0.3)
    if res.ok:
        summary = base + "/esummary.fcgi?db=gds&id=%s&retmode=json" % ",".join(res.json()["esearchresult"]["idlist"])
        r = requests.get(summary)
        time.sleep(0.3)
        if r.ok:
            for k,v in r.json()["result"].items():
                if not k == "uids":
                    if v["accession"] == geo_accession:
                        return v
        else:
            print(res.text)
            return False
    print(res.text)
    return False


def main(GSE_number):
    print(GSE_number)
    try:
        GEO_data = fetch_geo(GSE_number)
    except Exception as e:
        print("Error", e)
    df = check_sample_title(GEO_data)
    df = get_sample_landing_page(df)
    df = df.reset_index()
    del df['title']
    dic_df = df.to_dict(orient="index")
    return (jsonify(dic_df))


if __name__ == "__main__":
    # web server
    app = connexion.App(__name__)
    CORS(app.app)
    app.add_api(os.path.join(PTH,'swagger.yaml'))
    application = app.app
    app.run(port=8080, server='gevent')

# http://localhost:8080/searchtool?GSE_number=GSE112538
# cd /users/alon/desktop/github/geo
