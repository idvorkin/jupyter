# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore Customer Data
#   Should be applicable to all customer data sets,
#   Explores categories as well as time series
#

# %%
"""
from sklearn.datasets import fetch_openml
from sklearn import datasets, svm, metrics
from pandas import DataFrame
import matplotlib as mpl
"""

from typing import List, Tuple

# from dataclasses import dataclass
import pandas as pd

# import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# import glob
# import os
from pathlib import Path

# get nltk and corpus
# import nltk
# from nltk.corpus import stopwords

# get scapy and corpus
# import spacy
# import time
# from functools import lru_cache
import seaborn as sns
import humanize

# import swifter
import dask
import dask.dataframe as dd
from IPython.display import display

from humanize import intcomma, intword
import pandas_profiling
import arrow
import datetime
from functools import partial
from pandas_util import time_it


# %%
# make the plot wider
height_in_inches = 10
matplotlib.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# %% [markdown]
# # Load Data set

# %%
#raw_csv = "~/data/wamd.all.csv"
raw_csv = "/Users/idvorkin/imessage/all.messages.csv"
cleaned_df_pickled = f"{raw_csv}.pickle.gz"


# Load+Clean+Explore data using Dask as it's got multi-core.
# Then convert to a pandas dataframe pickle.
# df = dd.read_csv(raw_csv,sep='\t' )
# df = df.compute()
# df = pd.read_csv(raw_csv,sep='\t')
# df = pd.read_csv(raw_csv, sep="|", lineterminator="\n")
# df = pd.read_csv(raw_csv, sep="\t")
df


# %%
# clean up some  data

datetimeColumnName, customerIdColumnName = "date_uct", "id"

# setup date column
df["datetime"] = pd.to_datetime(df[datetimeColumnName], errors="coerce")
df = df.set_index(df.datetime)
# setup customer id
df["customer_id"] = df[customerIdColumnName]

# %%
# df = df.compute()
# df.to_pickle(cleaned_df_pickled)
ti = time_it(f"Load dataframe:{cleaned_df_pickled}")
df = pd.read_pickle(cleaned_df_pickled)
ti.stop()

# %%
# df = df.compute()
# df.to_pickle(cleaned_df_pickled)
# %%
# df = df.compute()
# df.to_pickle(cleaned_df_pickled)
# df.set_index
# df = df.reset_index(drop=True)
# kdf.index

# %% [markdown]
# # Data Analysis -
# ### Home Grown

# %%
# gotta be a more elegant way, but doing this for now

ti = time_it("compute distribution")
distribs = [df[c].value_counts(normalize=True).toPercent() for c in df.columns]
ti.stop()

# %%
def isFlatDistribution(d):
    return len(d) == 0 or d.iloc[0] < 0.01


for d in sorted(
    [d for d in distribs if not isFlatDistribution(d)], key=lambda d: d.iloc[0] * -1
):
    column_header = f"\n------ {d.name} ----- "
    print(column_header)
    print(f"{d.head(10)}")

print("++Flat distribution++")
for d in sorted([d for d in distribs if isFlatDistribution(d)], key=lambda d: d.name):
    c = d.name
    print(c)
print("--Flat distribution--")

# %% [markdown]
# # Data Analysis -
# ### Like the grown ups do.

# %%
# This looks pretty crappy on a black background, maybe change color first
# Also, I've had trouble with date indexes, might need to drop the index
# df = df.reset_index(drop = True)
profile_filename = "output.html"
ti = time_it(f"profiling dataframe to {profile_filename}")
#pr = df.reset_index(drop=True).profile_report()
#pr.to_file(output_file=profile_filename)
#pr
ti.stop()

# %% [markdown]
# # Time series analysis

# %%
# Any time series data interesting beyond count(), perhaps figure out pivots?
# https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
sns.set(rc={"figure.figsize": (11, 4)})
count_hourly = df.resample("M").count()
count_hourly.iloc[:, 0].plot(title="Interactions over time")


# %% [markdown]
# # Customer Distribution Analysis

# %%
def plot_distribution_for(df, count_buckets, minimum_call_count=0):
    cid = "customer_id"
    usage_by_cid = df[cid].value_counts()
    cid_to_exclude = usage_by_cid[
        usage_by_cid.values <= minimum_call_count
    ].index.values
    df = df[~df.customer_id.isin(cid_to_exclude)]
    usage_by_cid = df[cid].value_counts()
    dfT = usage_by_cid.value_counts(normalize=True).toPercent().iloc[:count_buckets]
    dfT.index.name = f"% usages by customer"
    N = len(usage_by_cid.value_counts())
    graphed = int(dfT.sum())
    title = "What % of time do customers call K times? "
    sub_title = f"Universe customers calling >= {minimum_call_count} times, N={humanize.intcomma(N)}, Visible={graphed}%"
    ax = dfT.plot(kind="bar", title=f"{title}\n{sub_title}")
    ax.set_ylabel(f"% customers")
    plt.show()  # force the plot ot show


# create a df w/only multi users for faster analysis
def df_w_multi_customers(df):
    usage_by_cid = df.customer_id.value_counts()
    cid_to_exclude = usage_by_cid[usage_by_cid.values == 1].index.values
    return df[~df.customer_id.isin(cid_to_exclude)]


df_multi = df_w_multi_customers(df)

ti = time_it("Plotting distributions")
plot_distribution_for(df, 3, 0)
plot_distribution_for(df, 10, 3)
plot_distribution_for(df, 10, 10)
plot_distribution_for(df, 10, 500)
ti.stop()

# %% [markdown]
# # MAU/WAU/DAU analysis


# %%
freq = "M"
cid_by_date = df["2019":].pivot_table(
    values="datetime",
    index=["customer_id"],
    columns=pd.Grouper(freq=freq),
    aggfunc="any",
)
# cid_sorted_by_sum = cid_by_date.T.sum().sort_values(ascending=False).index
# cid_by_date = cid_by_date.sort_values("customer_id")
# cid_by_date.T[cid_sorted_by_sum[4:5]].plot()
# cid_by_date.T.count().sort_values()
# df_hc.pivot_table(index=["customer_id"], columns=pd.Grouper(freq=freq), aggfunc="count")[ "customer_id" ].T.plot(title=f"customer {irange} by freq={freq}", figsize=(12, 8))


#%%
def cid_by_freq(df, freq):
    return df["2019":].pivot_table(
        values="datetime",
        index=["customer_id"],
        columns=pd.Grouper(freq=freq),
        aggfunc="any",
    )


# t = cid_by_freq(df, "M")


#%%
# cid_by_date.T.sum().sort_values(ascending=False)
def print_freq(df, freq, cuteName, minUsage):
    # minUsage should be inferred
    c = cid_by_freq(df, freq).sum(axis="columns")
    print(f"{cuteName}:{intcomma(len(c[c >= minUsage ]))}")


# NOTE: Could speed up significantly by removing
# Single time users
df_multi = df_w_multi_customers(df)
print(f"N:{intcomma(len(df.customer_id.value_counts()))}")
print(f"N(>1):{intcomma(len(df_multi.customer_id.value_counts()))}")
print_freq(df_multi, "M", "MAU_2", 2)
print_freq(df_multi, "M", "MAU", 5)
print_freq(df_multi, "W", "WAU", 20)
print_freq(df_multi, "D", "DAU", 140)


#%%
t.T.sum().sort_values(ascending=False)

#%%


#%%

# %%
customer_by_count = df.customer_id.value_counts()
print(f"customer_by_count\n{customer_by_count.head(10)}")
# customer_by_count =  customer_by_count
# df.obf_customer_id.value_counts().head(20)
start_range = 0
irange = range(start_range, start_range + 100)
print(f"customer_in_range\n{customer_by_count.iloc[irange]}")

df_hc = df[df.customer_id.isin(customer_by_count.index[irange].values)]
# count_hourly = df_hc.customer_id.resample("W").count()
# count_hourly = df_hc['2019-01':'2019-05'][["customer_id"]].groupby("customer_id").resample('D').count()
count_hourly
# count_hourly.plot()

## TODO: Head customer behavior
## TODO Middle
## TODO: Tail removal

## Look at head vs tail

# See step functions Called O-10
# Top 10 customers,
# trange = range(0, 3)
# customer_by_count.value_counts(normalize=True).apply(lambda d:d).iloc[trange] # .plot(kind='pie', title=f'% customer {trange}')


# %%
# pd.pivot_table?
