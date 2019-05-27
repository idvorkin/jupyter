# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python [Root]
#     language: python
#     name: Python [Root]
# ---

# +
# Count the number of ways a kid that can take 1, 2 or 3 steps can make 
# to cover N steps.


def CountSteps(n):
    if n < 1: return 0 
    if n == 1: return 1
    if n == 2: return 2
    if n == 3: return 4
    
    prev = (4,2,1)
    for i in range(3,n): # HYM, Why do I think it should be off by one?
        c = sum(prev) 
        prev = (c,prev[0], prev[1]) # shift down
    return prev[0]
for i in [1,3,7,36]: print ("Steps {}: Ways {}".format(i, CountSteps(i)))
# -

print ("{}",11)


# +
def CountMakeChangeExternal(denoms,total):
    if total <= 0: return 0
    if not denoms: return 0
    
    sorted = denoms.copy()
    sorted.sort(reverse=True)
    return CountMakeChange(sorted,total)
    
    
def CountMakeChange(denoms, total):
    assert(denoms)
    workDenoms = denoms.copy()
    currentCoin = workDenoms.pop() #pop mutates current.
    
    if total < 0: return 0
    if len(workDenoms) == 0:
        return 1 if total% currentCoin == 0 else 0 
            
    maxCoin = int(total/currentCoin)
    return sum ([CountMakeChange(workDenoms,total-i*currentCoin) for i in range (maxCoin+1)])


# -

CountMakeChangeExternal([5,3],14 )

CountMakeChangeExternal([25,10,5,1],80 )


def combsN(xs,n):
    if n == 0: yield []
    for (i,x) in enumerate(xs):
        # combine x with all combLenM1 not containing x's processed so Far.
        # all xs below i, are already in the list.
        yield from [[x] + ct for ct in combsN(xs[i+1:],n-1)]
        


for x in (combsN([1,2,3,4],4)): print (x)

# +
import itertools
def NQueensE(N):
    board = list(itertools.repeat(None,N))
    yield from NQueens(board,0)
    
def NQueens(board,toPlace):
    if lost(board): return 
    # print (board,toPlace)
    allPlaced = toPlace == len(board)
    if allPlaced: 
        yield board
        return 
    for (col,row) in enumerate(board):
        alreadySet = row !=None
        if alreadySet:continue
        copy = board[:]    
        copy[col] = toPlace
        yield from NQueens(copy,toPlace+1)
        
def lost(board):
    for (col, row) in enumerate(board):
        notPlaced = row == None
        if notPlaced: continue
        for (i, rowToCheck) in enumerate(board[col+1:]): 
            offsetKill = i+1
            if rowToCheck == row+offsetKill or rowToCheck == row-offsetKill:
                return True
    return False        
            
            
# -

len(list(NQueensE(8)))


# +
def genBalancedE(n):
    yield from genBalanced("",0,n,n)
    
def genBalanced(partial,cUnMatchedOpen,rOpen,rClosed):
    #print (partial,cUnMatchedOpen,rOpen,rClosed)
    full = (rOpen + rClosed) == 0
    if full:
        yield partial
        return 
    # always place an open if available
    if rOpen > 0:
        yield from genBalanced(partial+"(", cUnMatchedOpen+1, rOpen-1,rClosed)
        
    # place a close IF there's a close to match
    if cUnMatchedOpen > 0:
        yield from genBalanced(partial+")", cUnMatchedOpen-1, rOpen,rClosed-1)
        
        
# -

print (list (genBalancedE(5)))


# +
def decomposePalindrome(s):
    if len(s) == 0:  yield []; return
    for (i,x) in enumerate(s):
        head = s[:i+1]
        if isPalindrome(head): 
            yield from [[head] + subPal for subPal in decomposePalindrome(s[i+1:])]
            
def isPalindrome(s):
    for start,end,_ in zip(s, reversed(s), range(int(len(s)/2))): 
        if start != end: return False 
    return True

print (isPalindrome("ab"))
print (isPalindrome("a"))
print (isPalindrome("ab"))
print (isPalindrome("bb"))
print ("----")
print (list(decomposePalindrome("abba")))
print (list(decomposePalindrome("abbac")))
print ("----")
print (list(decomposePalindrome("a")))
print ("----")
print (list(decomposePalindrome("ab")))
print ("----")
print (list(decomposePalindrome("ecc")))
    
    
    

# +
def between (t,s,e): return t >= s and t <=e
def isRotated (s,e): return s > e

def onLeft (xs,t,s,m,e):
    vS = xs[s]
    vE = xs[e]
    vM = xs[m]
    
    # no rotation on left
    if not isRotated(vS,vM): return between(t, vS, vM)
        
    # no rotation on the right
    return not between(t,vM,vE)

def findRotatedE(xs,t): return findRotated(xs,t,0,len(xs)-1)
def findRotated(xs,t,s,e):
    noElements = s > e
    if noElements: return None
    iMid = int((s+e)/2)
    vMid = xs[iMid]
    if vMid == t: return iMid
    if onLeft(xs,t,s,iMid,e): 
        return findRotated(xs,t,s,iMid-1)
    else:
        return findRotated(xs,t,iMid+1,e)


# -

xs = [99,0,15,23,40,98]
findRotatedE(xs,40)






