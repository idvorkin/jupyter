jupytext --from ipynb --pipe black --pre-commit && jupytext --from ipynb --to py_generated//py:light --pre-commit
git diff --staged py_generated/*
