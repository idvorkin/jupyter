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

from bs4 import BeautifulSoup, NavigableString

header = """<h3> hello </h3> 
            <h1>hello <b>bob</b></h1> 
            <h1> hi <blockquote>there</blockquote> joey </h1>"""
soup = BeautifulSoup(header)


# +
def dump(el, indent=""):
    for c in list(el):
        isString = isinstance(c, NavigableString)
        if isString:
            print(f"{indent}STR:{c.strip()}")
        else:
            print(
                f"{indent}Node:{c.name}, Children:{len(list(c.children))}:{indent}{c}"
            )
            dump(c, indent + " ")


dump(soup)
# -




