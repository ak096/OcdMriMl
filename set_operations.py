from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def subsequentset(iterable):
    sset = []
    s = list(iterable)
    for idx, i in enumerate(s):
        sset.append(s[0:idx+1])
    return sset

