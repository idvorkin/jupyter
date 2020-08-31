# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Pick your project
#

# +
# %pylab inline
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn3_unweighted
import numpy as np
import io

plt.xkcd()
plt.title("Allocate Resources by \n Maximizing Overlap")
out = venn3(
    subsets=(10, 10, 10, 10, 9, 4, 3),
    set_labels=("Personal Preference", "Business Need", "Probability Of Success"),
    alpha=0.5,
)
# out = venn3_unweighted(subsets = (20, 10, 12, 10, 9, 4, 3), set_labels = ("Personal Preference", "Business Need", "Likely To Succeed"), alpha = 0.5);

for idx, subset in enumerate(out.subset_labels):
    out.subset_labels[idx].set_visible(False)

plt.show(out)
# -


# _The irony that I've written this entire post using complex technology is not lost on me. This post was to satisfy my own 'technological itch'_

buf = io.BytesIO()
plt.savefig(buf, format="svg")
buf.seek(0)

buf


