[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/idvorkin/jupyter)

# jupyter

A place for my jupyter notebooks, see them [live](http://nbviewer.jupyter.org/github/idvorkin/jupyter).

## Good plugins

```
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyterlab_vim
jupyter labextension install jupyterlab-jupytext

```

## Authoring Tips

- [XKCD style graphs in matplotlib](http://nbviewer.jupyter.org/url/jakevdp.github.io/downloads/notebooks/XKCD_sketch_path.ipynb)

## Vim usage jupyter-lab

V2 of the jupyter shell will be jupyter-lab. They have a seperate VIM plugin:
https://github.com/jwkvam/jupyterlab_vim

## Jupyter diff tools

[nbdiff](https://github.com/jupyter/nbdime#installation) - Cool tool for graphical diff of notebooks.

## Jupyter To Python Round Trip ( JupyText)

https://github.com/mwouts/jupytext

Use From the command line:

    # Roundtrip and execute via black
    jupytext --sync --pipe black PlayNLP.ipynb

## Vim usage

- Useful Keyboard shortcuts
  - Tab to tab-complete
  - C-G to see tool tips
- C-Enter -> execture cell.
- A-Enter -> execute cell and create cell below
- S-Esc -> Enter command mode.
- i -> insert mode in cell
- O -> Create cell above
- o -> create cell below

Use ssh to expose a remote server to localhost using ssh

    # Connections on 4444 will get redirected to 8888 on the remote_host.
    ssh -N -L localhost:8888:localhost:4444 remote_user@remote_host

Resize Jupyter to window width

    # Update container css
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))

    # Make matlab drawings wider
    height_in_inches=8
    mpl.rc("figure", figsize=(2*height_in_inches,height_in_inches))

Useful incantations \* Igor's [Pandas Tips](https://github.com/idvorkin/techdiary/blob/master/notes/pandas-tutorial.md)
