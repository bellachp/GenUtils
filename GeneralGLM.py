# glm stuff
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score


infile = "C:/projects/PotentiometerData.csv"
outfile = "/testing.csv"


results = []
volt = []
attn = []
bias = []

with open(infile) as f:
    csvReader = csv.reader(f, delimiter=",")
    ctr = 0
    for row in csvReader:
        if ctr == 0:
            print(row)
            ctr += 1
        else:
            results.append(row[0])
            volt.append(row[1])
            attn.append(row[2])
            bias.append(row[3])


data = np.column_stack((results, volt, attn, bias))

np.random.shuffle(data)

N = len(data)
trainN = 1000

trainingData = data[0:trainN,:]
testData = data[trainN:N, :]





#betaHat = np.linalg.lstsq(data, results)[0]
#print(betaHat)




# plt.figure(1)
# plt.scatter(data[:,0], data[:,1])
# plt.title("volt vs attn")
# plt.xlabel("volt")
# plt.ylabel("attn")

# plt.figure(2)
# plt.scatter(data[:,0], data[:,2])
# plt.title("volt vs bias")
# plt.xlabel("volt")
# plt.ylabel("bias")

# plt.figure(3)
# plt.scatter(data[:,1], data[:,2])
# plt.title("attn vs bias")
# plt.xlabel("attn")
# plt.ylabel("bias")


# plt.show()