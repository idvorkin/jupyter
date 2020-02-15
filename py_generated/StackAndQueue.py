# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + jupyter={"outputs_hidden": true}
from typing import List, Set, Dict, Tuple, Optional


class QueueFromStacks:
    def __init__(self):
        self.newOnTop: List[int] = []
        self.oldOnTop: List[int] = []

    def push(self, v):
        self.newOnTop.append(v)

    def pop(self):
        if len(self.oldOnTop) == 0:
            # take from new on top to build oldOnTop
            while len(self.newOnTop) != 0:
                self.oldOnTop.append(self.newOnTop.pop())
        return self.oldOnTop.pop()


# + jupyter={"outputs_hidden": false}
m = QueueFromStacks()
m.push(3)
m.push(4)
m.pop()
m.push(6)
m.pop()
m.pop()
m.push(8)
m.pop()
# -


