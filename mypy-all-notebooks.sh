# run jupyter
jupytext --from ipynb --to python//py:light *.ipynb

# run mypy (TBD into vim)

mypy --check-untyped-defs python/*py --ignore-missing-imports
