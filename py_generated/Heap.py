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

# +
from typing import List, Set, Dict, Tuple, Optional


def lhs(i):
    return 2 * i + 1


def rhs(i):
    return 2 * i + 2


def parent(i):
    return int((i - 2) / 2 if isEven(i) else (i - 1) / 2)


def isEven(i):
    return i % 2 == 0


def heapify_up(storage, lastElement=None):

    if not lastElement:
        lastElement = len(storage) - 1

    # push bottom element up as needed.
    i = lastElement

    while i != 0:
        if storage[i] <= storage[parent(i)]:
            return

        # swap and move to kids.
        (storage[i], storage[parent(i)]) = (storage[parent(i)], storage[i])
        i = parent(i)


def heapify_down(storage):
    # push first element down till we're a valid heap, or done.
    i = 0

    while i < len(storage):  # we're not in the end pos.
        # Watch to make sure we do the last iteration --

        validLHS = lhs(i) < len(storage)
        validRHS = rhs(i) < len(storage)

        isLeaf = not validRHS and not validLHS

        if isLeaf:
            return

        isOnlyLHS = not validRHS

        if isOnlyLHS:
            if storage[i] < storage[lhs(i)]:
                (storage[i], storage[lhs(i)]) = (storage[lhs(i)], storage[i])
            return  # no more nodes.

        # we have a RHS and a LHS swap with the biggest.
        vLHS = storage[lhs(i)]
        vRHS = storage[rhs(i)]
        iMax = lhs(i) if vLHS > vRHS else rhs(i)

        if storage[i] >= storage[iMax]:
            # we're already a heap
            return

        (storage[i], storage[iMax]) = (storage[iMax], storage[i])
        i = iMax


# todo parameterize min/max heap
class Heap:
    def __init__(self):
        self.storage: List[int] = []

    def last(self, i):
        len(self.storage) - 1

    def head(self):
        if not self.storage:
            return None
        return self.storage[0]

    def push(self, i):
        # add an element to the end
        self.storage.append(i)
        heapify_up(self.storage)

    def pop(self):
        # handle last element.
        # store first element
        ret = self.storage[0]
        # place last element in first slot
        last = self.storage.pop()
        if not self.storage:
            # last element popped
            return last

        self.storage[0] = last
        heapify_down(self.storage)
        return ret

    def __str__(self):
        return str(self.storage)


# + jupyter={"outputs_hidden": false}
import random

h = Heap()

for i in range(7):
    h.push(random.randint(-9999, 999))
for i in range(7):
    print(h.pop())

# + jupyter={"outputs_hidden": false}
r = [random.randint(0, 200) for i in range(15)]
print(r)
for i in range(len(r)):
    heapify_up(r, i)
z = Heap()
z.storage = r
for i in range(len(r)):
    print(z.pop())


# + jupyter={"outputs_hidden": true}


# + jupyter={"outputs_hidden": true}


# + jupyter={"outputs_hidden": true}

