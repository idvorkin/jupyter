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

# ## Igor's Evolution from Technologist to Entrepreneur
#
# **Entrepreneurs say** - What does my customer need? How do I make sure their needs are met?
#
# **Technologists say** - What's the most I can do with my technology? How do I push the limits of my technology?
#
# As I've gotten older, I've realized I can provide more value to world by focusing on customer problems.  Because of my long career as a technologist, I still love technology and ensure technologists focus the maximum effort on solving the customer need, and not  scratching their 'technological itch'.

# +
# %pylab inline
import matplotlib.pyplot as plt
import numpy as np

plt.xkcd()

start_year = 2002
end_year = 2018
interest_shift_year = 2014

years = np.linspace(start_year, end_year)
interest_entrepener = 2 + 98.0 / (1 + np.exp(0.6 * (interest_shift_year - years)))
interest_technologist = np.maximum(
    (100 - interest_entrepener), 50 + 10 * (np.sin(50 * years))
)
interest_entrepener = np.minimum(interest_entrepener, 70 + 10 * (np.sin(40 * years)))

plt.plot(years, interest_entrepener, "-b", label="Entrepreneur")
plt.plot(years, interest_technologist, "-r", label="Technologist")
plt.ylabel("Identity")
plt.xlabel("Year")

plt.xlim(start_year, end_year)
plt.legend(loc="best")
# -

# _The irony that I've written this entire post using complex technology is not lost on me. This post was to satisfy my own 'technological itch'_


