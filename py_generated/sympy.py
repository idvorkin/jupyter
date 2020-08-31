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

# +
from sympy import *
from sympy.plotting import plot

init_printing()
x, y, z = symbols("x y z")
# -


x + y * 2

sqrt(x)

z = x ** 2

z

# https://docs.sympy.org/latest/modules/plotting.html
plot(x + cos(x), x + 4, 10 * sqrt(x), (x, -20, 20))

z

y = x ** 2

plot(y + 5)


