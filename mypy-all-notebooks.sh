# run jupyter
jupytext --from ipynb --to py_generated//py:light *.ipynb

# run mypy (TBD into vim)

mypy --check-untyped-defs py_generated/*py --ignore-missing-imports
