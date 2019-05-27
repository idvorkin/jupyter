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
def reverse(s,start,end):
    while (start < end):
        # swap
        s[start], s[end]  = s[end],s[start]
        # move pointers in
        start=start+1
        end = end-1
        
def words(s):
    start = 0
    for i in range (len(s)):
        on_delimeter =  s[i] == " "
        if on_delimeter: 
            conscutiveSpace = i == start
            if conscutiveSpace: 
                start = start+1
                continue
                
            end = i-1 # letter before delim
            yield (start,end)
            start = i+1 # letter after delim
            
    lastDelimIsSpace = start == len(s)        
    if lastDelimIsSpace: 
        return
    
    yield (start,len(s)-1)

def reverseWords(s) : 
    for w in words(s): reverse(s,*w)
        
s=[*"  hi there  joe a b   "]
print  ("<start>"+''.join(s)+"<end>")
            
        
# -

import collections
Word = collections.namedtuple("Word",['start','end'])
w = Word(4,5)
print (*w)


def computeLongestFirstSubPaths(path):
    if not path or path[0] != '/': return None
    segments = path.split('/')
    print (segments)
    for iEnd in range(len(segments),0,-1):
        print (iEnd)
        yield segments[1:iEnd]



[p for p in computeLongestFirstSubPaths('/bob/barker')]

