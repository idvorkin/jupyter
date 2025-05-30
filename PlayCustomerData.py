# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# from dataclasses import dataclass
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# import glob
import os

# get nltk and corpus
# import nltk
# from nltk.corpus import stopwords

# get scapy and corpus
# import spacy
# import time
# from functools import lru_cache
import humanize

# import swifter
from IPython.display import display

from humanize import intcomma

# import pandas_profiling
import datetime
from datetime import timedelta, date
from functools import partial
from pandas_util import time_it
import altair as alt


# %%
# make the plot wider
height_in_inches = 10
matplotlib.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# %% [markdown]
# # Load Data set

# %%
# raw_csv = "~/data/wamd.all.csv"
# raw_csv = "~/data/imessage.csv"
raw_csv = "~/data/sergio.csv"
cleaned_df_pickled = f"{os.path.expanduser(raw_csv)}.pickle.gz"


# Load+Clean+Explore data using Dask as it's got multi-core.
# Then convert to a pandas dataframe pickle.
# dask_df = dd.read_csv(raw_csv,sep=',' )
# df = dask_df.compute()
df = pd.read_csv(raw_csv, sep=",")
# df = df.compute()
# df = pd.read_csv(raw_csv,sep='\t')
# df = pd.read_csv(raw_csv, sep="|", lineterminator="\n", error_bad_lines=False)
# df = pd.read_csv(raw_csv, sep="\t", lineterminator='\r"')
df


# %%
# df = df.compute()
# df.to_pickle(cleaned_df_pickled)
# ti = time_it(f"Load dataframe:{cleaned_df_pickled}")
# df = pd.read_pickle(cleaned_df_pickled)
# ti.stop()

# %%
isWorkChat = True
isImessage = False
# clean up some  data

datetimeColumnName, customerIdColumnName = "date_uct", "id"

if isWorkChat:
    datetimeColumnName, customerIdColumnName = "date", "name"

if isImessage:
    datetimeColumnName, customerIdColumnName = "date", "to_phone"


# setup date column
df["datetime"] = pd.to_datetime(df[datetimeColumnName], errors="coerce")


df["customer_id"] = df[customerIdColumnName]

# for workplace chat
if isWorkChat:
    df["is_from_me"] = df.apply(
        lambda r: r.customer_id == "Igor Dvorkin", axis=1
    )  # depends on
df = df.set_index(df.datetime)
df

# %% [markdown]
# # Data Analysis -
# ### Home Grown

# %%
# gotta be a more elegant way, but doing this for now

ti = time_it("compute distribution")
distribs = [df[c].value_counts(normalize=True) for c in df.columns]
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
# pr = df.reset_index(drop=True).profile_report()
# pr.to_file(output_file=profile_filename)
# pr
ti.stop()


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
    dfT.index.name = "% usages by customer"
    N = len(usage_by_cid.value_counts())
    graphed = int(dfT.sum())
    title = "What % of time do customers call K times? "
    sub_title = f"Universe customers calling >= {minimum_call_count} times, N={humanize.intcomma(N)}, Visible={graphed}%"
    ax = dfT.plot(kind="bar", title=f"{title}\n{sub_title}")
    ax.set_ylabel("% customers")
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


# %%
def cid_by_freq(df, freq):
    return df["2019":].pivot_table(
        values="datetime",
        index=["customer_id"],
        columns=pd.Grouper(freq=freq),
        aggfunc="any",
    )


# %%
# cid_by_date.T.sum().sort_values(ascending=False)
def print_freq(df, freq, cuteName, minUsage):
    # minUsage should be inferred
    c = cid_by_freq(df, freq).sum(axis="columns")
    print(f"{cuteName}:{intcomma(len(c[c >= minUsage]))}")


# NOTE: Could speed up significantly by removing
# Single time users
df_multi = df_w_multi_customers(df)
print(f"N:{intcomma(len(df.customer_id.value_counts()))}")
print(f"N(>1):{intcomma(len(df_multi.customer_id.value_counts()))}")
print_freq(df_multi, "M", "MAU_2", 2)
print_freq(df_multi, "M", "MAU", 5)
print_freq(df_multi, "W", "WAU", 20)
print_freq(df_multi, "D", "DAU", 140)


# %% [markdown]
# # Time Series Analysis

# %%
df.resample("M").count()["datetime"].plot(title="Total usage over time")
plt.show()

df_tr = df["2019"]
customer_by_count = df_tr.customer_id.value_counts()
irange = range(0, 5)
df_hc = df_tr[df_tr.customer_id.isin(customer_by_count.index[irange].values)]

top_customer_by_month = df_hc.pivot_table(
    values="datetime",
    index=["customer_id"],
    columns=pd.Grouper(freq="M"),
    aggfunc="count",
    fill_value=0,
).T

for kind in "line area bar".split():
    top_customer_by_month.plot(title="Top customers usage over time", kind=kind)
    plt.show()
print(f"customer_by_count\n{customer_by_count.head(10)}")


# %%
df


# %%
# df[df.customer_id=="Sergio Wolman"].datetime.dt.weekday.value_counts().orderby()
# df = df[df.customer_id=="Sergio Wolman"]

# %%


# %%
#  Resample to hours
# Select weekdays between 6 and 19
def filter_work_hours(df):
    df = by_hour_count(df)
    return df[(df.index.hour > 5) & (df.index.hour < 19) & (df.index.weekday < 5)]


def by_hour_count(df):
    return df.resample("H").count().rename(columns={"customer_id": "count_messages"})


def pivot_day_by_hour_no_melt(df, agg):
    df = df.resample("H").count().rename(columns={"customer_id": "count_messages"})
    pv = df.pivot_table(
        index=df.index.weekday,
        columns=df.index.hour,
        values="count_messages",
        aggfunc=agg,
    )
    pv.index.name, pv.columns.name = "day", "hour"
    return pv


def pivot_day_by_hour(df, agg):
    pv = df.pivot_table(
        index=df.index.weekday,
        columns=df.index.hour,
        values="count_messages",
        aggfunc=agg,
    )
    pv.index.name, pv.columns.name = "day", "hour"
    return pv


def heat_map(df, agg, title, is_pivoted=False):
    m = df if is_pivoted else pivot_day_by_hour(df, agg)
    m = m.reset_index().melt(id_vars=["day"])

    # turn pivotted day/hour back to datetimes for altair plotting
    start_of_week = date.today() - timedelta(date.today().weekday())
    now = datetime.datetime(year=2000, month=1, day=1)
    m.day = pd.to_datetime(m.day.apply(lambda x: start_of_week + timedelta(days=x)))
    m.hour = pd.to_datetime(m.hour.apply(lambda x: now + timedelta(hours=now.hour + x)))

    hm = (
        alt.Chart(m)
        .mark_rect()
        .encode(alt.Y("day(day):O"), alt.X("hours(hour):O"), color=alt.Color("value"))
        .properties(title=title)
    )

    text = (
        alt.Chart(m)
        .mark_text()
        .encode(
            alt.Y("day(day):O"),
            alt.X("hours(hour):O"),
            alt.Text("value", format="d"),  # format as decimal
            color=alt.value("white"),
        )
        .properties(title=title)
    )

    return hm + text


# hm =  draw_heat_map(t2,functools.partial(np.quantile,q=quantile),f"{int(quantile*100)}th percentile")
sergio = df[df.name == "Sergio Wolman"]
# heat_map(filter_work_hours(sergio), np.count, "Sergio Time Count")
# display(hm)

# from_ammon =  df[(df.customer_id=='+12063567091') & (df.is_from_me == False)]
# to_ammon =  df[(df.customer_id=='+12063567091') & (df.is_from_me == True)]
for q in [60, 80, 95, 99]:
    agg = partial(np.quantile, q=q * 0.01)
    # f_ammon = pivot_day_by_hour_no_melt(from_ammon,agg)
    # t_ammon = pivot_day_by_hour_no_melt(to_ammon,agg)
    # delta_weight = t_ammon/(f_ammon+t_ammon).apply(lambda x:x*.01)
    # display(heat_map(delta_weight,agg=None, is_pivoted=True, title=f"% Igor Messages P{q}"))
    hm = heat_map(by_hour_count(sergio), agg, f"Ammon P{q}")
    # display(hm)

for q in [50, 90, 95, 99]:
    agg = partial(np.quantile, q=q * 0.01)
    # f_ammon = pivot_day_by_hour_no_melt(from_ammon,agg)
    # t_ammon = pivot_day_by_hour_no_melt(to_ammon,agg)
    # delta_weight = t_ammon/(f_ammon+t_ammon).apply(lambda x:x*.01)
    # display(heat_map(delta_weight,agg=None, is_pivoted=True, title=f"% Igor Messages P{q}"))
    hm = heat_map(filter_work_hours(sergio), agg, f"Sergio<->Igor chat messages @ P{q}")
    display(hm)


# %%
