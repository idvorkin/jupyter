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
def Primes(n):
    if not n or n < 1:
        raise Exception("n must be positive", n)
    possible = [i + 1 for i in range(n)]
    nextPrime = 2
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2]
    while nextPrime:
        nextPrime = crossOff(possible, nextPrime)
    return [i for i in possible if i]


def crossOff(possible, prime):
    nextPrime = None
    for i in range(prime, len(possible)):
        if possible[i] % prime == 0:
            possible[i] = 0
        if possible[i] and not nextPrime:
            nextPrime = possible[i]
    return nextPrime


# + jupyter={"outputs_hidden": false}
Primes(1)
# -

Primes(3)
Primes(18)

# + jupyter={"outputs_hidden": false}
Primes(-1)

# + jupyter={"outputs_hidden": true}

