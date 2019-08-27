# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.0-rc1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore NLP against my journal entries
#
# This notebook allows me to play with NLP concepts using my personal journals. 
# I've been writing personal journal entries ala 750 words a day for several years. 

# %%
"""
from sklearn.datasets import fetch_openml
from sklearn import datasets, svm, metrics
from pandas import DataFrame
import matplotlib as mpl
"""

from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

import glob
import os
from pathlib import Path

# get nltk and corpus
import nltk
from nltk.corpus import stopwords

# get scapy and corpus
import spacy
import time
from functools import lru_cache
from pandas_util import time_it
from matplotlib import animation, rc
from IPython.display import HTML
from datetime import timedelta
import itertools 

# %%
# This function is in the first block so you don't
# recreate it willy nilly, as it includes a cache.

nltk.download("stopwords")

# Remove domain words that don't help analysis.
# Should be factored out
domain_stop_words = set(
    """
    yes yup Affirmations get that's Journal
    Deliberate Disciplined Daily
    Know Essential Provide Context
    First Understand Appreciate
    Hymn Monday
    """.lower().split()
)


@lru_cache(maxsize=4)
def get_nlp_model(model: str):
    start_time = time.time()
    print(f"Loading Model {model}")
    nlp = spacy.load(model)  # python -m spacy download en_core_web_lg
    spacy.prefer_gpu()  # This will be cool if/when it happens.
    duration = time.time() - start_time
    print(f"Took: {int(duration)}")
    return nlp


# Load corpus of my daily ramblings
@dataclass(frozen=True)
class Corpus:
    path: str
    all_content: str
    initial_words: List[str]
    words: List[str]

    def __hash__(self):
        return self.path.__hash__()


@lru_cache(maxsize=100)
def LoadCorpus(corpus_path: str) -> Corpus:

    # Hym consider memoizing this asweel..
    english_stop_words = set(stopwords.words("english"))
    all_stop_words = domain_stop_words | english_stop_words

    corpus_path_expanded = os.path.expanduser(corpus_path)
    corpus_files = glob.glob(corpus_path_expanded)

    """
    ######################################################
    # Performance side-bar.
    ######################################################

    A] Below code results in all strings Loaded into memory for temporary,  then merged into a second string.
    aka Memory = O(2*file_conent) and CPU O(2*file_content)

    B] An alternative is to do += on a string results in a new memory allocation and copy.
    aka Memory = O(file_content) , CPU O(files*file_content)

    However, this stuff needs to be measured, as it's also a funtion of GC. Not in the GC versions there is no change in CPU
    Eg.

    For A] if GC happens after every "join", then were down to O(file_content).
    For B] if no GC, then it's still O(2*file_content)
    """

    # Make single string from all the file contents.
    list_file_content = [Path(file_name).read_text() for file_name in corpus_files]
    all_file_content = " ".join(list_file_content)

    # NOTE I can Upper case Magic to make it a proper noun and see how it ranks!

    properNouns = "zach ammon Tori amelia josh Ray javier Neha Amazon John".split()

    def capitalizeProperNouns(s: str):
        for noun in properNouns:
            noun = noun.lower()
            properNoun = noun[0].upper() + noun[1:]
            s = s.replace(" " + noun, " " + properNoun)
        return s

    # Grr- Typo replacer needs to be lexically smart - sigh
    typos = [
        ("waht", "what"),
        ('I"ll', "I'll"),
        ("that", "that"),
        ("taht", "that"),
        ("ti ", " it "),
        ("that'sa", "that's a"),
        ("Undersatnd", "Understand"),
        ("Ill", "I'll"),
        ("Noitce", "Notice"),
        ("Whcih", "Which"),
        ("K ", "OK "),
        ("sTories", "stories"),
        ("htat", "that"),
        ("Getitng", "Getting"),
        ("Essenital", "Essential"),
        ("whcih", "which"),
    ]

    def fixTypos(s: str):
        for typo in typos:
            s = s.replace(" " + typo[0], " " + typo[1])
        return s

    all_file_content = fixTypos(all_file_content)
    all_file_content = capitalizeProperNouns(all_file_content)

    # Clean out some punctuation (although does that mess up stemming later??)
    initial_words = all_file_content.replace(",", " ").replace(".", " ").split()

    words = [word for word in initial_words if word.lower() not in all_stop_words]
    return Corpus(
        path=corpus_path,
        all_content=all_file_content,
        initial_words=initial_words,
        words=words,
    )


@lru_cache(maxsize=100)
def DocForCorpus(nlp, corpus: Corpus):
    print(
        f"initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}"
    )
    ti = time_it(f"Building corpus from {corpus.path} of len:{len(corpus.all_content)} ")
    # We use all_file_content not initial_words because we want to keep punctuation.
    doc_all = nlp(corpus.all_content)

    # Remove domain specific stop words.
    doc = [token for token in doc_all if token.text.lower() not in domain_stop_words]
    ti.stop()
    
    return doc


# %% [markdown]
# # Build corpus from my journal in igor2/750words

# %%
# make function path for y/m


# Build up corpus reader.
# Current design, hard code knowledge of where everything is stored in forwards direction.
# Hard code insufficient data for analysis.
# Better model, read all files in paths and then do lookup.



# Hymn better model: 
# A - lookup all files. 
# B - Generate paths based on actual locations.

def glob750_latest(year, month):
    assert month in range(1, 13)
    base750 = "~/gits/igor2/750words/"
    return f"{base750}/{year}-{month:02}-*.md"

def glob750_new_archive(year, month):
    assert month in range(1, 13)
    base750 = "~/gits/igor2/750words_new_archive/"
    return f"{base750}/{year}-{month:02}-*.md"


def glob750_old_archive(year, month):
    assert month in range(1, 13)
    base750archive = "~/gits/igor2/750words_archive/"
    return f"{base750archive}/750 Words-export-{year}-{month:02}-01.txt"


def corpus_paths_months_for_year(year):
    return [glob750_old_archive(year, month) for month in range(1, 13)]


# Corpus in "old archieve"  from 2012-2017.
corpus_path_months = {
    year: corpus_paths_months_for_year(year) for year in range(2012, 2018)
}

# 2018 Changes from old archive to new_archieve.
# 2018 Jan/Feb/October don't have enough data for analysis
corpus_path_months[2018] = [glob750_old_archive(2018, month) for month in range(3, 8)] + [
    glob750_new_archive(2018, month) for month in (9, 11, 12)
]

corpus_path_months[2019] = [glob750_new_archive(2019, month) for month in range(1, 8)]+ [glob750(2019, month) for month in range(8, 9)]

corpus_path_months_trailing = [
    glob750(2018, month) for month in (9, 11, 12)
] + corpus_path_months[2019]


# TODO: Add a pass to remove things with insufficient words.


# %%
# make the plot wider
height_in_inches = 8
matplotlib.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# %% [markdown]
# ### Load simple corpus for my journal

# %%
corpus = LoadCorpus(corpus_path_months[2019][0])
print(f"initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}")


# %%
# Could use nltk frequency distribution plot, but better off building our own.
# fd = nltk.FreqDist(words)
# fd.plot(50, percents=True)
# Can also use scikit learn CountVectorizor

# %%
# Same as NLTK FreqDist, except normalized, includes cumsum, and colors
def GraphWordDistribution(words, title="", skip=0, length=50, includeCDF=True) -> None:
    def GetPDFCDF(words):
        def ToPercent(x: float) -> float:
            return x * 100

        # NOTE: No point creating a full data frame when only using a single column.
        pdf = pd.Series(words).value_counts(normalize=True).apply(ToPercent)
        cdf = pdf.cumsum()
        return (pdf, cdf)

    def PlotOnAxis(series, ax, label: str, color: str):
        # RANT: Why is MPL so confusing? The OO interface vs the stateful interface, GRAH!!
        # The random non-obvious calls.
        # GRAH!!!

        ax.legend(label.split())
        ax.plot(series, color=color)

        # RANT: Why no YAxis.set_labal_params()? E.g.
        #                 ax.yaxis.set_label_params(label, color=color)
        ax.set_ylabel(label, color=color)
        ax.yaxis.set_tick_params(labelcolor=color)

        # technically all the X axis paramaters are duplicated since we "twinned the X paramater"
        ax.xticks = range(len(series))

        # RANT: rot can be set on plt.plot(), but not on axes.plot()
        ax.xaxis.set_tick_params(rotation=90)

    # NOTE: can make graph prettier with styles E.g.
    # with plt.style.context("ggplot"):
    fig, ax = plt.subplots(1)

    ax.set_title(title)
    ax.grid(True)

    # make pdf first axes, and cdf second axes.
    ax_pdf, ax_cdf = (ax, ax.twinx())
    color_pdf, color_cdf = ("green", "blue")
    pdf, cdf = GetPDFCDF(words)

    PlotOnAxis(pdf[skip : skip + length], ax_pdf, label="PDF*100", color=color_pdf)
    PlotOnAxis(cdf[skip : skip + length], ax_cdf, label="CDF*100", color=color_cdf)


GraphWordDistribution(corpus.words, title="Normalized Word Distribution")

# %%
skip = 10
GraphWordDistribution(
    corpus.words, skip=skip, length=75, title=f"Distribution without top {skip} words"
)

# %%
# wordcloud is non-deterministic, which is bizarre.
# from wordcloud import WordCloud
# wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=None).generate("".join(words))
# plt.imshow(wordcloud,  interpolation='bilinear')

# %% [markdown]
# # Play with POS tagging and lemmatisation

# %%
nlp = get_nlp_model("en_core_web_lg")
nlp.max_length = 100 * 1000 * 1000


def GetInterestingWords(pos: str, doc, corpus: Corpus):
    interesting_pos = pos
    interesting_pos_set = set(interesting_pos.split())
    interesting = [token for token in doc if token.pos_ in interesting_pos_set]
    interesting_words = [token.lemma_ for token in interesting]
    return interesting_words


def GraphPoSForDoc(pos: str, doc, corpus):
    GraphWordDistribution(
        GetInterestingWords(pos, doc, corpus=corpus),
        title=f"Distribution of {pos} on {corpus.path}",
        skip=0,
        length=20,
    )


def GraphScratchForCorpus(corpus_path: str, pos: str = "NOUN VERB ADJ ADV"):
    corpus = LoadCorpus(corpus_path)
    doc = DocForCorpus(nlp, corpus)
    GraphPoSForDoc(pos, doc, corpus)


def GetInterestingForCorpusPath(corpus_path: str, pos: str = "NOUN VERB ADJ ADV"):
    corpus = LoadCorpus(corpus_path)
    doc = DocForCorpus(nlp, corpus)
    return GetInterestingWords(pos, doc, corpus)


# %%
# corpus_paths = corpus_paths_years
corpus_paths = corpus_path_months[2016]
print(corpus_paths)
# %%
for c in corpus_paths:
    GraphScratchForCorpus(c, pos="PROPN")

# %% [markdown]
# # Debugging when stuff goes goofy.

# %%
_ = """
max_to_analyze = 15
interesting = [token for token in doc if token.tag_ == "NNS"]
for token in interesting[:max_to_analyze]:
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)

# Parts of speech: https://spacy.io/usage/linguistic-features
GraphWordDistribution([token.pos_ for token in doc], title=f"POS Distribution on {corpus_path}")
# interesting = [ token for token in doc if token.pos_ != "PUNCT" and token.pos_ != "SYM" and len(token.text) > 3]
"""


# %% [markdown]
# ### Visualizing the "Thought Distribution" over time.
# * A] Sentiment over time. Graph valence as line graph time series
#     (TBD: Use cloud service to analyze each file)
#
# * B] Graph a bar chart of Proper noun trending over time, have it update per corpus file.
#  * Build a data frame of word frequency "Proper Noun"x"Corpus"
#  * Graph update every second.

# %%
def MakePDF(words, name):
    def ToPercent(x: float) -> float:
        return x * 100

    return pd.Series(words, name=name).value_counts(normalize=True).apply(ToPercent)


def PathToFriendlyTitle(path: str):
    path = path.split("/")[-1]
    if "export-" in path:
        return path.split("export-")[-1]
    else:
        return path


# %%
# corpus_paths = corpus_path_months[2018]+corpus_path_months[2019]
corpus_paths = corpus_path_months[2018] + corpus_path_months[2019]
print(corpus_paths)
pdfs = [
    MakePDF(GetInterestingForCorpusPath(p, "PROPN"), PathToFriendlyTitle(p))
    for p in corpus_paths
]

# TODO: Why can't we use the join - gives an error.
# wordByTimespan = pd.DataFrame().join(pdfs, how="outer", sort=False)
wordByTimespan = pd.DataFrame()
for pdf in pdfs:
    wordByTimespan = wordByTimespan.join(pdf, how="outer")

# Sort by sum(word frequency) over all corpus
# I  suspect it'd be interesting to sort by TF*IDF because it'll make words that are present
# only in a few months get a boost.
wordByTimespan["word_frequency"] = wordByTimespan.sum(skipna=True, axis="columns")
wordByTimespan = wordByTimespan.sort_values("word_frequency", ascending=False)


# Remove total column
wordByTimespan = wordByTimespan.iloc[:, :-1]

top_words_to_skip,   count_words   = 5, 10
print (f"skipping:{top_words_to_skip}, count:{count_words} ")

# wordByTimespan.iloc[:50, :].plot( kind="bar", subplots=False, legend=False, figsize=(15, 14), sharey=True )
wordByTimespan.iloc[top_words_to_skip:top_words_to_skip + count_words, :].T.plot(
    kind="bar", subplots=True, legend=False, figsize=(15, 9), sharey=True
)
# wordByTimespan.iloc[:13, :].T.plot( kind="bar", subplots=False, legend=True, figsize=(15, 14), sharey=True )

# %%
top_word_by_year = wordByTimespan.iloc[:15,:][::-1] # the -1 on the end reverse the count

anim_fig_size=(16,10)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = top_word_by_year.iloc[:,0].plot(
    title=f"Title Over Written", figsize=anim_fig_size,  kind='barh'
)

animation.patches = ax.patches
loop_colors = itertools.cycle('bgrcmk')
animation.colors = list(itertools.islice(loop_colors,len(animation.patches)))


def animate(i, ):
    # OMG: That was impossible to find!!!
    # Turns out every time you call plot, more patches (bars) are added to graph.  You need to remove them, which is very non-obvious.
    # https://stackoverflow.com/questions/49791848/matplotlib-remove-all-patches-from-figure
    [p.remove() for p in reversed(animation.patches)] 
    top_word_by_year.iloc[:,i].plot(title=f"Distribution {top_word_by_year.columns[i]}", kind='barh', color=animation.colors , xlim=(0,10))
    return (animation.patches,)


anim = animation.FuncAnimation(
    fig, animate, frames=len(top_word_by_year.columns), interval=timedelta(seconds=1).seconds * 1000, blit=False 
)
HTML(anim.to_html5_video())

# %%
dmo = """
corpus_path = "~/gits/igor2/750words/2019-06-*md"
corpus = LoadCorpus(corpus_path)
doc = DocForCorpus(nlp, corpus)
for t in doc[400:600]:
print(f"{t} {t.lemma_} {t.pos_}")
"""
from spacy import displacy

displacy.render(nlp("Igor wonders if Ray is working too much"))

# %%
corpus_path_months

# %%
corpus_path_months_trailing



# %%
