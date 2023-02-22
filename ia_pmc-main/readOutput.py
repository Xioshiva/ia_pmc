import operator
import matplotlib.pyplot as plt
import numpy as np

f = open('outPMC.txt', 'r')
r = []
data = []
lines = f.readlines()
for line in lines:
        line = line.rstrip("\n")
        r.append(line.split(' '))
        tmp = []
        tmp.append(int(r[-1][-5][11:]))
        tmp.append(int(r[-1][-4][6:]))
        tmp.append(float(r[-1][-1][6:12]))
        data.append(tmp)
f.close()
s = sorted(data, key = operator.itemgetter(1,0))
sortedData = []
for i in range(36):
    tmp = []
    for j in range(10):
        tmp.append(s.pop(0))
    sortedData.append(tmp)
listSTD = []
listAVG = []
listPARAM = []
for i in sortedData:
    listSTD.append(np.std([100 - x[2] for x in i]))
    listAVG.append(np.mean([100 - x[2] for x in i]))
    listPARAM.append("n=%d i=%d" %(i[0][0],i[0][1]))
plt.xlabel("Success Rate average and standard deviation in black")
plt.title("10 fold cross validation of bands.dat") 
plt.barh(listPARAM,listAVG, xerr= listSTD)
plt.show()
