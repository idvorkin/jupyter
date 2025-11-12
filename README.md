# My random jupyter notebooks - to be updated

A place for my jupyter notebooks, see them [live](http://nbviewer.jupyter.org/github/idvorkin/jupyter) here.

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/idvorkin/jupyter?urlpath=lab)

## Interactive Demos

**[Weight Analysis (Marimo WASM)](https://weight-analysis.surge.sh)** - Interactive weight tracking and visualization running entirely in your browser with Python WASM. Features matplotlib boxplots, Altair charts, and animated weight progression.

## Most important

I think, most important, editting in VS.Code is now good enough - w00t! (Not sure what plugins is making it so)

## Setup envinroment

- See how I install/setup pyproject.toml
- Also see the .pre-commit.yaml

## Good plugins

- BEST Plugin: VSCode/Cursor
- https://github.com/jupyterlab-contrib/jupyterlab-vim
- https://github.com/jupyter/nbdime

```bash
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyterlab-jupytext
```

### CPU limiting

Sometimes jupyter will take all the CPU and drop my ssh session.
Install cpulimit and run jupyter lab with it

    cpulimit -l 90 jupyter lab

### Variable Inspector

[https://github.com/lckr/jupyterlab-variableInspector]

## Authoring Tips

- [XKCD style graphs in matplotlib](http://nbviewer.jupyter.org/url/jakevdp.github.io/downloads/notebooks/XKCD_sketch_path.ipynb)

## Jupyter diff tools

[nbdime/nbdif](https://github.com/jupyter/nbdime#installation) - Cool tool for graphical diff of notebooks.

Often I just want to see the code that changed:

```zsh
alias nbdiffcode="nbdiff --ignore-metadata --ignore-details --ignore-output"
```

## Jupyter To Python Round Trip ( JupyText)

<https://github.com/mwouts/jupytext>

Use From the command line:

    # Roundtrip and execute via black
    jupytext --sync --pipe black PlayNLP.ipynb

## Vim usage in jupyter lab

- Useful Keyboard shortcuts
  - Tab to tab-complete
  - C-G to see tool tips
- C-Enter -> execture cell.
- A-Enter -> execute cell and create cell below
- S-Esc -> Enter command mode.
- i -> insert mode in cell
- O -> Create cell above
- o -> create cell below

Add this to your VIM config, to let fj be escape

```json
{
  "enabled": true,
  "extraKeybindings": [
    {
      "context": "insert",
      "mapfn": "map",
      "enabled": true,
      "command": "fj",
      "keys": "<Esc>"
    },
    {
      "command": "fk",
      "keys": "<Shift-Esc>",
      "context": "insert"
    }
  ],
  "enabledInEditors": true
}
```

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
