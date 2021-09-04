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

# + [markdown] toc-hr-collapsed=false
# # BSTs
#
# Notes:
#
# - Confirm how they want to do duplicates (inline with tree, or count on node)
#
# ### Can't end recursion with just the current knowledge.
#
# Recursion where you just return state of final node tends not to be the solution (as it is with trees)
# This is because you usually want some idea of parent state (see idea of a candidate node) which is easier to do with a while loop with a candidate node and bi-secting search space each time.
#
# ### Complexity
#
# O(n) if need to visit all
# O(h) if height of tree
# O(h) can degrade to o(n) as can always see a BST as an LL
#
#
# ### Approaches:
#
# **Abstract out traversal (usually in order)**
#
# Differentiate iteration, from conditions. Iteration is an o(n), depending on starting point.
#
# **K-largest:** Traverse BST backwards by doing inorder rhs->mid->lhs, stop after K.
#
# InOrderBackwards(tree).first(5) # use a generator, take first 5 or none - very tight syntax
#
# **Pass valid intervals down:**
#
# * IsBST() => InRange(t,min,max)
#
# **Keep track of candidate node, while cutting search space in half:**
#
# * NextBiggest()
# * FirstFound()
#
# Can use recursion,but a bit easier to read with iteration
# Note this is the same as operating in a sorted array
#
# candidate = None; it = t
#
# while (it):
#  if isFound: return True
#  if onLeft: it = it.lhs
#  if onRight: it = it.rhs
# return False
#
# ### Analog to sorted array
#
# - You cut candidates in half each time
# - None/Terminal node => e>s
# - lhs = (low, mid-1)
# - rhs = (mid+1, high)
# - mid == (s+e)//2
#

# +
class Tree:
    def __init__(self, i, lhs=None, rhs=None):
        self.value = i
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"[{self.value}], [{self.lhs}], [{self.rhs}]"

    def __repr__(self):
        return str(self)


def printTree(tree, depth=0):
    if not tree:
        return
    printTree(tree.lhs, depth + 1)
    print(f'{" " * depth * 4}[{tree.value}]')
    printTree(tree.rhs, depth + 1)


# + jupyter={"outputs_hidden": false}
printTree(None)
printTree(Tree(4))
print("--BST--")
bst1 = Tree(4, Tree(3), Tree(6, Tree(5), Tree(7)))
printTree(bst1)
notbst3 = Tree(4, Tree(3), Tree(6, Tree(9), Tree(7)))
print("--Not BST--")
printTree(notbst3)

# + jupyter={"outputs_hidden": false}


# + jupyter={"outputs_hidden": false}
def InOrderTraverse(tree):
    if not tree:
        return
    yield from InOrderTraverse(tree.lhs)
    yield tree
    yield from InOrderTraverse(tree.rhs)


def IsBst(tree):
    prev = None
    for it in InOrderTraverse(tree):
        firstElement = prev == None
        if firstElement:
            prev = it
            continue  # first element always a BST
        if prev.value > it.value:
            return False
        prev = it
    return True


# + jupyter={"outputs_hidden": false}
print([n.value for n in InOrderTraverse(bst1)])
IsBst(bst1)

# + jupyter={"outputs_hidden": false}
print([n.value for n in InOrderTraverse(notbst3)])
IsBst(notbst3)


# + jupyter={"outputs_hidden": false}
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


# + jupyter={"outputs_hidden": false}
FindFlip([2, 1])

# + jupyter={"outputs_hidden": true}
FindFlip([9, 9, 1, 1])

# + jupyter={"outputs_hidden": true}


# + jupyter={"outputs_hidden": true}

# -


