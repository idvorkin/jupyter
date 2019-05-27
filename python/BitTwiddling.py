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

def printB(x):
    print("{0}:{0:b}".format(x))


printB(4)
printB(19)
printB(-1)
printB(~0)
printB(16)
printB(15)

printB((1 << 5) - 1)  # set low order bits, over shift, -1.
# -1 on a power of 2 flips all the bits that are lower.

# set high order bits needs negation, which is hard to do since no maximum width.
x = ((1 << 32) - 1) << 4
printB(x)


def InsertBits(src, dst, start, end):
    maskUpper = ((1 << (32)) - 1) << (end + 1)
    printB(maskUpper)
    maskLower = (1 << (start)) - 1
    printB(maskLower)
    mask = maskUpper | maskLower
    printB(mask)
    maskedDst = dst & mask
    shiftedSrc = src << start
    return maskedDst | shiftedSrc


# TODO Review for correctness :(
m = InsertBits(0, 127, 3, 6)
printB(m)


def PrintDoubleAsString(d):
    if d >= 1 or d < 0:
        return None  # Note 0 is a special case, not handled.
    bitValue = 1.0  # set to 1 so first time through loop gets correct value
    remaining = d
    representation = ""
    for bitPost in range(32):
        if not remaining:
            break
        bitValue = bitValue / 2
        if remaining >= bitValue:
            bitRepresentation = "1"
            remaining -= bitValue
        else:
            bitRepresentation = "0"

        representation = bitRepresentation + representation

    return None if remaining else representation


PrintDoubleAsString(0.5)

PrintDoubleAsString(0.125)
PrintDoubleAsString(0.625)

PrintDoubleAsString(0.24)


