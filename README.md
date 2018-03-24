[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/idvorkin/jupyter)

# jupyter
A place for my jupyter notebooks, see them [live](http://nbviewer.jupyter.org/github/idvorkin/jupyter).

## Authoring Tips
* [XKCD style graphs in matplotlib](http://nbviewer.jupyter.org/url/jakevdp.github.io/downloads/notebooks/XKCD_sketch_path.ipynb)

## Vim usage jupyter-lab
V2 of the jupyter shell will be jupyter-lab. They have a seperate VIM plugin:
https://github.com/jwkvam/jupyterlab_vim

installation is easy: 

    jupyter labextension install jupyterlab_vim

## Vim usage
* Install from [here](https://github.com/lambdalisue/jupyter-vim-binding)
* Useful Keyboard shortcuts
    * Tab to tab-complete
    * C-G to see tool tips
* C-Enter -> execture cell.
* A-Enter -> execute cell and create cell below
* S-Esc -> Enter command mode.
* i -> insert mode in cell
* O -> Create cell above
* o -> create cell below

Temporary Install of Vim plugin

    %%javascript
    Jupyter.utils.load_extensions('vim_binding/vim_binding');p


## Install/Setup
* Install via Anaconda
* Run in anaconda window: Jupyter notebook

Use ssh to expose a remote server to localhost using ssh

    # Connections on 8889 will get redirected to 8888 on the remote_host.
    ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host

Resize Jupyter to window width

    # Update container css
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))

    # Make matlab drawings wider
    height_in_inches=8
    mpl.rc("figure", figsize=(2*height_in_inches,height_in_inches))
