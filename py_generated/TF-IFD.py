# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Play TF/IFD

# +
from sklearn.datasets import fetch_openml
from sklearn import datasets, svm, metrics
from pandas import DataFrame
import matplotlib as mpl

import glob
import os
from pathlib import Path

# +
corpus_paths = ["~/gits/igor2/750words/2019*md",
                "~/gits/igor2/750words/2018*md",
                "~/gits/igor2/750words_archive/*2012*txt",
                "~/gits/igor2/750words_archive/*2013*txt",
                "~/gits/igor2/750words_archive/*2014*txt",
                "~/gits/igor2/750words_archive/*2015*txt",
                "~/gits/igor2/750words_archive/*2016*txt",
                "~/gits/igor2/750words_archive/*2017*txt",
                "~/gits/igor2/750words_archive/*2018*txt",
                "~/gits/igor2/750words/2019-01-*md",
                "~/gits/igor2/750words/2019-02-*md",
                "~/gits/igor2/750words/2019-03-*md",
                "~/gits/igor2/750words/2019-04-*md",
                "~/gits/igor2/750words/2019-05*md",
                "~/gits/igor2/750words/2019-06-*md",
               ]
    
def path_glob_to_string_of_words(path):
    path_expanded = os.path.expanduser(path)
    files = glob.glob(path_expanded)
    # Make single string from all the file contents.
    list_file_content = [Path(file).read_text() for file in files]
    all_file_content = " ".join(list_file_content)
    return all_file_content
    

corpus = [ path_glob_to_string_of_words(p) for p in corpus_paths]
len(corpus)

# +
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# I can skip stop words because I'm going to use TF/IDF
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
# -

assert X.get_shape()[0] == len(corpus_paths), "There should be a row per corpus path"
feature_labels = vectorizer.get_feature_names()
# Should be a column per word. aka Huge!

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
Y = transformer.fit_transform(X)
DataFrame(transformer.idf_,index=feature_labels)[0].sort_values(ascending=True)

from pandas import DataFrame 
# At this point, every column is the IF/TDF of the words in the document. 
# In theory the largest elements of the array would be the biggests.
assert len(feature_labels) == Y[0].get_shape()[1] , "Should have a laber for each column "

df = DataFrame(Y.toarray().transpose(), index=feature_labels, columns=corpus_paths)

# +

range(10)[-8:-4]
# -

start = 20
for x in df.columns:
     print(df[x].sort_values(ascending=False)[start:start+100])
#df.iloc[:,1].sort_values(ascending=False)[50:100]

# +
# Dataframe().join?

# +
# DataFrame().join?

# +
d = DataFrame()
# d.sum?
# -


