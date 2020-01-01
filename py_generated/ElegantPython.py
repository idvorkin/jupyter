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

# ### Enumerate
# Iterate w/index
#

# enumerate w/index
for x, i in enumerate("hello"):
    print(x, i)

# ### Default dictionary
# A dictionary which generates value when doesn't exit

# +
from collections import defaultdict

key = "arbitrary_key"
d_set = defaultdict(set)
d_set[key].add("word")
print(d_set[key])

d_int = defaultdict(int)
d_int[key] += 1
d_int[key] += 1
print(d_int[key])
# -

# ### List Manipulation

# +
l = [i for i in range(3)]
print(l)

# # + and += only operate on enumerables (not singleton), but it's tight syntax to do a single e.g. l+=[one]
# Use l+[] to extend by a list
# Use l+= [] to modify in place

print("you can join lists using the + operator, but it needs a list on both sides.")
print(l + [1])
print(l)
print("you can extend in place using += ")
l += ["Igor"]
print(l)

print("you can also in place using append and extend.")
l.append("Bob")
print(l)
l.extend(["Ammon"])
print(l)
# -

# ### Dictionary Manipulation

# +
d = {}
d["k1"] = "v1"
d["k2"] = "v2"

print("enumerate keys and values")
for k, v in d:
    print(k, v)

print("enumerate keys")
for k in d.keys():
    print(k)

print("enumerate values")
for v in d.values():
    print(v)
# -


# ### Peek Iterator.
# Often you need to walk a list and keep track of index, this isn't a real iterator but you can make one.

# +
from itertools import chain


class PeekIterator:
    class EndOfIteration:
        pass

    def __init__(self, iter):
        self.iter = chain(iter, [self.EndOfIteration])
        self.next = next(self.iter)

    def peek(self):
        # shouldn't be allowed
        # throw
        if self.is_empty():
            raise StopIteration
        return self.next

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.is_empty():
                break
            r = self.next
            self.next = next(self.iter)
            return r
        raise StopIteration

    # Increment iterator and return value
    def is_empty(self):
        return self.next == self.EndOfIteration

    def has_elements(self):
        return not self.is_empty()

    def pop(self):
        return next(self)


# +
z1 = PeekIterator(range(10))
z1.peek()
z1.peek()

# Iteration works normally. Now lets do manual iteration.
for i in PeekIterator(range(10)):
    print(i)

i1 = PeekIterator(range(4))

while i1.has_elements():
    print(f"Peek:{i1.peek()}")
    print(f"Pop:{i1.pop()}")
    print(f"IsEmpty:{i1.is_empty()}")
    print(f"has_elements:{i1.has_elements()}")


# +
print("Use it to do a merge of 2 sorted lists")
import random

l1 = [random.randint(0, 100) for x in range(10)]
l1.sort()
print(l1)

l2 = [random.randint(0, 100) for x in range(10)]
l2.sort()
print(l2)


def merge(xs, ys):
    pxs = PeekIterator(xs)
    pys = PeekIterator(ys)
    out = []

    while pxs.has_elements() and pys.has_elements():
        itSmaller = pxs if pxs.peek() < pys.peek() else pys
        out += [itSmaller.pop()]

    remaining = pxs if pys.is_empty() else pys
    out += pxs
    return out


print(merge(l1, l2))


# + [markdown] toc-hr-collapsed=true
# ### Intervals and sorting and named tupples
# For interval questions, normally you do sort endpoints and go from there.

# +
from collections import namedtuple


def maxAlive(bds):
    Index = namedtuple("Index", "index increment")
    print(bds)
    indexs = []
    for b, d in bds:
        indexs += [Index(b, increment=1)]
        indexs += [Index(d, increment=-1)]

    indexs = sorted(indexs, key=lambda x: x.index)
    print(indexs)

    cAlive = 0
    maxAlive = 0
    for i in indexs:
        cAlive += i.increment
        maxAlive = max(maxAlive, cAlive)
    return maxAlive


birth_deaths = [(4, 10), (9, 12), (20, 25), (22, 28), (19, 40)]
print(maxAlive(birth_deaths))
# -

# ### Set operations

# +
print("can set via list comprehension")
a = {x for x in "qabracadabra" if x not in "abc"}
b = set("abracadabra")
c = set("alacazam")

print(f"a:{a}")
print(f"b:{b}")
print(f"b-a:{b-a}")
print(f"Union A|B:{a|b}")
print(f"Intersect (subset) A&B:{a&b}")
print(f"Only in 1 (XOR) (A|B - A&B) A^B:{a^b}")
# -


# ### Stack and Queue
# Stack is LIFO (Simple List)
# Queue is FIFO (Special Class)
# Deque is dual sided O(1) access, pop(),popleft(), append(), appendleft()

# +
# S
l = []
l.append(4)  # Add to End
l.append(5)  # Add to End

# monkey patch top, pretty syntax
print(f"stack.top(see last element, L[-1]): {l[-1]}")
print(f"stack.pop(remove from end): {l.pop()}")

from collections import deque

q = deque(["Eric", "John", "Michael"])
print(f"Q:{q}")
print(f"Q.popleft():{q.popleft()}")
print(f"Q.pop():{q.pop()}")

q.append("Igor")
q.appendleft("hi")
print(f"Q:{q}")
# -

# ### Enums

# +
from enum import Enum, auto


class Color(Enum):
    RED = auto()
    BLUE = auto()
    GREEN = auto()


z = Color.RED
print(z)

# also support bit flags to perform set operations, but heck, just use set operations it's clearer.
# -

# # Generator Funtions
# * Return values using a simple function
# * Generators are lazy, so don't maintain extra memory
# * Need to make them a list using list()
# * Great to make Iterators from logic (e.g. Tree in-order traversal, LL traversal etc)
# * Great to make sequences of infinite length e.g. nextInt() as they don't require building a list.
# * Nice solution when you find yourself appending to a variable to return like:
# * ret = []... ret+=[next value] return ret
#
# Syntax:
# * return -> Done
# * yield x -> return value x
# * yield from xs -> return values as passed in.

# +
def gen3(start):
    yield start
    yield start + 1
    yield start + 2
    return


def gen4(start):
    yield start
    yield from gen3(start + 1)
    return


print(list(gen3(0)))
print(list(gen4(0)))
