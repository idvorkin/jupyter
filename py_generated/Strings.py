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

# + jupyter={"outputs_hidden": false}
def reverse(s, start, end):
    while start < end:
        # swap
        s[start], s[end] = s[end], s[start]
        # move pointers in
        start = start + 1
        end = end - 1


def words(s):
    start = 0
    for i in range(len(s)):
        on_delimeter = s[i] == " "
        if on_delimeter:
            conscutiveSpace = i == start
            if conscutiveSpace:
                start = start + 1
                continue

            end = i - 1  # letter before delim
            yield (start, end)
            start = i + 1  # letter after delim

    lastDelimIsSpace = start == len(s)
    if lastDelimIsSpace:
        return

    yield (start, len(s) - 1)


def reverseWords(s):
    for w in words(s):
        reverse(s, *w)


s = [*"  hi there  joe a b   "]
print("<start>" + "".join(s) + "<end>")


# + jupyter={"outputs_hidden": false}
import collections

Word = collections.namedtuple("Word", ["start", "end"])
w = Word(4, 5)
print(*w)


# + jupyter={"outputs_hidden": false}
def computeLongestFirstSubPaths(path):
    if not path or path[0] != "/":
        return None
    segments = path.split("/")
    print(segments)
    for iEnd in range(len(segments), 0, -1):
        print(iEnd)
        yield segments[1:iEnd]


# + jupyter={"outputs_hidden": false}
[p for p in computeLongestFirstSubPaths("/bob/barker")]

# + jupyter={"outputs_hidden": true}

