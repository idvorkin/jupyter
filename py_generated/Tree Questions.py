# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

print(3 + 4)


class Tree:
    def __init__(self, i, lhs=None, rhs=None):
        self.value = i
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "[{0}][L:{1}][R:{2}]".format(self.value, self.lhs, self.rhs)

    def __repr__(self):
        return str(self)


def printTree(tree, depth=0):
    if not tree:
        return
    printTree(tree.lhs, depth + 1)
    print("{0}[{1}]".format(" " * depth * 4, tree.value))
    printTree(tree.rhs, depth + 1)


printTree(None)

printTree(Tree(4))

bst1 = Tree(4, Tree(3), Tree(6, Tree(5), Tree(7)))
printTree(bst1)
notbst3 = Tree(4, Tree(3), Tree(6, Tree(9), Tree(7)))
print("---")
printTree(notbst3)


# + {"active": ""}
#
# -

def IsBst(tree, last=0):
    if not tree:
        return (True, last)
    (isBst, last) = IsBst(tree.lhs, last)
    if not isBst:
        return (False, last)
    if last and last > tree.value:
        return (False, last)
    return IsBst(tree.rhs, tree.value)


def InOrderTraverse(tree):
    if not tree:
        return
    rhs = tree.rhs  # making safe for mutation for bi-note solution
    yield from InOrderTraverse(tree.lhs)
    yield tree.value
    yield from InOrderTraverse(rhs)


[n for n in InOrderTraverse(bst1)]

IsBst(bst1)

IsBst(notbst3)


# +
def FindFlip(a):
    return FindFlipR(a, 0, len(a) - 1)


def FindFlipR(a, start, end):
    length = end - start + 1
    if length == 1:
        return None
    if length == 2:
        if a[start] > a[end]:
            return a[end]
        else:
            return None
    mid = start + int((end - start) / 2)
    print(">", start, mid, end, a[start], a[mid], a[end])

    if a[start] < a[mid]:
        # it's on the right
        return FindFlipR(a, mid, end)
    if a[start] > a[mid]:
        # it's on the left
        return FindFlipR(a, start, mid)
    return None


# -

FindFlip([2, 1])

FindFlip([9, 9, 1, 1])






