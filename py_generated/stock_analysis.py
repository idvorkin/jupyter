# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import matplotlib

matplotlib.style.use("ggplot")
from ipywidgets import interact
import numpy as np
import matplotlib as mpl
import arrow
from matplotlib import animation, rc
from IPython.display import HTML, display
from datetime import timedelta
import pandas_datareader.data as data


# %matplotlib inline
# -

# # Lets take a stab at some stock analysis!

# +
start_date = "2017-06-01"  # SNAP  IPO'd on 3/2/2017 - so start after that.
end_date = "2020-12-31"

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(
    "FB;AMZN;SNAP;AAPL;QQQ;GOOG;MSFT;GOLD".split(";"), "yahoo", start_date, end_date
)
print("Sample from the full dataset")
display(panel_data.head(1))
df_original = panel_data["Close"]
print("Sample of data from close")
df = df_original.copy()
display(df.head(3))

# +
from numpy.lib.function_base import disp
import arrow

earliest = arrow.utcnow().shift(months=-12).date()
df = df_original.copy()[
    earliest:
]  # note this is destructive, probably good to keep an original around as a best practice

df.index = df.index.astype(
    str
)  # when transposing dates to columns, easier to operate in strings.

first_day = df.reset_index().iloc[0, 0]
last_day = df.reset_index().iloc[-1, 0]

display(f"Returns from:{first_day}, to:{last_day}")

returns = df.iloc[
    [0, -1]
].T  # first and last row, and turn into columns for easy manipulation
returns["delta"] = returns[last_day] - returns[first_day]
returns["pcnt_change"] = returns.delta / returns[first_day]
display(returns)

# returns.diff = returns.[] _[0] - _[1]
print(
    "QQ: Should sum daily %% change ==  pcnt_change of total returns -- because it doesn't??"
)
df.pct_change().sum()


# +
from numpy.core.defchararray import encode
from altair.vegalite.v4.schema.channels import Tooltip

print("Correlations between stocks")
print("  NOTE: Need to correlate on percentage change, not abosolute price")


def display_correlation_matrix(df, earliest, latest):
    df = df_original.copy()
    df.columns.name = None
    corr = df.pct_change(1).corr()  # compute correlation on percent change
    df_dates = df[earliest:latest]
    corr = df_dates.corr().reset_index().melt(id_vars="index")
    # display(corr)
    height_in_inches = 40

    base = (
        alt.Chart(corr)
        .properties(
            width=8 * height_in_inches,
            height=8 * height_in_inches,
            title=f"Stock Correlation from: {earliest} to {latest}",
        )
        .encode(
            x="index:O",
            y="variable:O",
        )
    )

    c = base.mark_rect().encode(
        tooltip="value;index;variable".split(";"), color="value:Q"
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("value:Q", format="0.2f")
    )
    display(c + text)


# Look at correlation
year_ago = arrow.utcnow().shift(months=-12).date()
today = arrow.utcnow().date()
latest = today
cv_19_start = "2020-02-01"

import datetime

first_day = df_original.reset_index().iloc[0, 0]

display_correlation_matrix(df, year_ago, today)
display_correlation_matrix(df, first_day, today)
display_correlation_matrix(df, "2020-06-01", today)
display_correlation_matrix(df, "2019-01-01", cv_19_start)
display_correlation_matrix(df, cv_19_start, today)


# + active=""
#

# + active=""
#
