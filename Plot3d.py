# simple 3d plotting
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from sklearn.cluster import KMeans

metaFile = "C:/projects/smartmap/volmeta.csv"
volFile = "C:/projects/smartmap/volpoints.csv"

hardList = [
(35.5165404, -17.57840909, -33.28270202), #
#(11.61676449, -19.94130776, -28.05406299),
(29.56453212, 3.66980571, -30.28191911), #
#(14.78032566, 2.18720493, 12.31881684),
(-31.38746097, -13.50468412, -27.66826808), #
#(-2.21908333, -3.933, -21.78241667),
(29.67092181, -33.85675709, -18.72848901), #
(12.57961731, -24.15668884, 2.71511867), #
#(12.35911462, 19.15914006, -17.09680702),
(-28.70012346, -31.35160494, -11.29160494)], #
#(31.5729629, -7.7747204, -1.90537902),
#(-3.94193964, -24.84449761, -12.23573795)]


basePt = (-1.0, -1.0, -1.0)
baseVec = (-1.0, -1.0, -1.0)
volArray = []


#meta
with open(metaFile, "r") as f:
    ctr = 0
    for line in f:
        pt = line.rstrip().split(",")
        if ctr == 0:
            ctr += 1
            basePt = (float(pt[0]), float(pt[1]), float(pt[2]))
        else:
            baseVec = (float(pt[0]), float(pt[1]), float(pt[2]))

basePtstep = (basePt[0] + 10.0 * baseVec[0], basePt[1] + 10.0 * baseVec[1], basePt[2] + 10.0 * baseVec[2])

#voldata
with open(volFile, "r") as f:
    for line in f:
        row = line.rstrip().split(",")
        pt = (float(row[0]), float(row[1]), float(row[2]))
        if pt != (0.0, 0.0, 0.0):            
            volArray.append(pt)


print(len(volArray))

data = np.array(volArray)

np.random.shuffle(data)

cut = int(len(data) * 0.02)
subd = data[0:cut, :]
print(np.shape(subd))

# cluster on big batch
start = time.time()
clust = KMeans(n_clusters=12)
clust.fit(data)
labels = clust.labels_[0:cut]
clusters = clust.cluster_centers_
end = time.time()
print("big job done " + str(end - start))


fig = plt.figure(1)
ax = Axes3D(fig)
ax.scatter(subd[:,0], subd[:,1], subd[:,2], 
    c=labels.astype(np.float), edgecolor='k')
ax.scatter(basePt[0], basePt[1], basePt[2], color="red", alpha=0.5, edgecolor='k', s=70)
ax.scatter(basePtstep[0], basePtstep[1], basePtstep[2], color="red", edgecolor='k', s=70)


fig = plt.figure(2)
ax = Axes3D(fig)
ax.scatter(clusters[:,0], clusters[:,1], clusters[:,2], edgecolor='k')
ax.scatter(basePt[0], basePt[1], basePt[2], color="red", alpha=0.5, edgecolor='k', s=70)
ax.scatter(basePtstep[0], basePtstep[1], basePtstep[2], color="red", edgecolor='k', s=70)
#testClust = np.array(hardList)
#ax.scatter(testClust[:,0], testClust[:,1], testClust[:,2], color="orange")


print(clusters)

print("testing")
testClust = np.array(hardList)
for pt in testClust:
    print(pt - basePt)


plt.show()
