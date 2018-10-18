import pdb
import numpy as np

def import_batch(fname = '', n = 0):
    batchList = []
    fdata = open(fname)
    for line in fdata:
        parts = line.split()
        lineDict = {}
        lineDict['label'] = 1.0 if parts.pop(0) == '+1' else -1.0
        lineDict['dataArray'] = np.zeros((1, n), dtype='f4')
        for part in parts:
            key, val = part.split(':')
            lineDict['dataArray'][0, int(key) - 1] = float(val)
        batchList.append(lineDict)
    return batchList 