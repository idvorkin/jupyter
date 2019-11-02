# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Enumerate
# Iterate w/index
#

# enumerate w/index
for x, i in enumerate("hello"):
    print(x, i)

# ### Default dictionary
# A dictionary which generates value when doesn't exit

# +
from collections import defaultdict

key = "arbitrary_key"
d_set = defaultdict(set)
d_set[key].add("word")
print(d_set[key])

d_int = defaultdict(int)
d_int[key] += 1
d_int[key] += 1
print(d_int[key])
