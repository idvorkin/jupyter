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

# +
# TODO: Still a bug in here
def unsortedMin(xs, i, iMin):
    assert i > 0
    x = xs[i]
    if not iMin:
        iMin = i - 1

    assert xs[iMin] != x

    if x > xs[iMin]:
        return (iMin, xs[iMin])

    for i in range(iMin, 0, -1):
        isSorted = x >= xs[i]
        if isSorted:
            return (i + 1, x)
    return (0, x)


def unsortedRange(xs):
    if not xs:
        return
    if len(xs) == 1:
        return
    maxValue = xs[0]
    minValue = xs[0]
    i = 1
    iUnsortedMin = None
    iUnsortedMax = None
    for i in range(1, len(xs)):
        x = xs[i]
        inOrder = x >= maxValue
        if inOrder:
            maxValue = x
        else:
            (iUnsortedMin, minValue) = unsortedMin(xs, i, iUnsortedMin)
            iUnsortedMax = i
            print(i, iUnsortedMin, iUnsortedMax)
            assert iUnsortedMin < i

    allSorted = iUnsortedMin == None or iUnsortedMax == None
    if allSorted:
        assert iUnsortedMin == None or iUnsortedMax == None
        return None
    return (iUnsortedMin, iUnsortedMax)


# -

unsortedRange([-3, 1, 0])


def plusOne(ds):
    for i in range(len(ds) - 1, -1, -1):
        d = ds[i]
        print((i, d))
        if d < 9:
            ds[i] = ds[i] + 1
            return
        if d == 9:
            if i == 0:
                ds[0] = 1
                ds.append(0)
            else:
                ds[i] = 0


d = [1, 9, 3, 9, 9]
plusOne(d)
print(d)
