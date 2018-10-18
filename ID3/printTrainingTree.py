import sys
import ID3
import data

# Data provided by class instructor, obtained from the liblinear repository, heavily modified for class purposes (the original data will not work with this code)
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms

# Set up arguments for ID3 Algorithm
# trainingData = data.Data(fpath='experiment-data/data/train.csv')
attributes = dict(trainingData.column_index_dict)
del attributes['label']

#Run ID3 Algorithm
ID3Tree = ID3.ID3(trainingData, attributes, "root")
print("Final Tree:")
print(ID3Tree)