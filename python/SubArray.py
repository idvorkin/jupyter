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
def len1d(x): return x[1] - x[0]
def smallest(x,y):
    if not x: return y
    if not y: return x
    return x if len1d(x) < len1d(y) else y
    
    
print (len1d((3,4)))
print (smallest((100,200), (10,20)))
print (smallest(None, (10,20)))

# +
from itertools import * 

def getRange(d): 
    t = list(d.values())
    if [1 for x in t if x is None]: return None
    t.sort()
    r= (t[0], t[-1])
    print (r)
    return r

def minSubArray(xs,vals):
    lastSeenIndex = dict(zip(vals,repeat(None)))
    for val in vals: lastSeen[val] = None
    
    minRange = None
    for (i,x) in enumerate(xs): 
        if x in lastSeenIndex:
            lastSeenIndex[x] = i
            minRange =  smallest(getRange(lastSeenIndex), minRange)
    return minRange        
            
        
    
# -

xs = list("aaaacmmmmadaaaabffffffffffcmm3134daffffffffffffecdefaddd")
ss = list("acd")
print (minSubArray(xs,ss))


