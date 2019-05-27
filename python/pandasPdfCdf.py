# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import mpld3
# %matplotlib inline
# mpld3.enable_notebook() # All mpld3 gives is pan and zoom, which isn't great. 
# Maybe when it gets more plugins I'll reconsider it
# 

# +
rand = np.random.randint(100,200, size=30309)
df = pd.DataFrame(rand)

bins = [0,150,160,165,170,180,199,200]
countByBin =df.groupby(pd.cut(df[0], bins)).count()
countByBin["normalized"] = countByBin*100.0/countByBin.sum()
countByBin["cumsum"] = countByBin["normalized"].cumsum()
# -

fig, axs = plt.subplots(ncols=len(countByBin.columns), nrows=1)
for iCol in range(3) : 
    countByBin.iloc[:,iCol].plot(kind="bar", title=f"{countByBin.columns[iCol]}",ax=axs[iCol])

# +
#countByBin["normalized;cumsum".split(";")].plot(kind="bar")
fig = plt.figure()
ax = countByBin["normalized"].plot(kind="bar")
ax.plot(countByBin["cumsum"].values, color='red', marker="o") #TBD have one one scale

# To have seperate y axis, start by twinning the axis, and then 
# plotting on the twin  - e.g.
# ax2 = ax.twinx()
# ax2.plot (...)
ax.set_title("PDF & CDF")

# It'd be great if these plugins work - but they don't seem to.
# tt1 = mpld3.plugins.LineLabelTooltip(ax, label='l1')
# mpld3.plugins.connect(fig, tt1)

# -

# normalized
# pdf
sns.distplot(df)

# cdf
sns.distplot(df, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))

plt.hist(df)
plt.title("Non normalized distribution")


