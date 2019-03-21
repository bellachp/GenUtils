# SurfacePatch.py

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
import scipy.special
from scipy import ndimage
import time


def ComputeBernsteinPoly(n, k, u):
    bernCoeff = scipy.special.binom(n, k) * u**k * (1-u)**(n-k)
    return bernCoeff

#bezier poly_mat (cubic/quadratic based on order)
#only in z term
def ComputeBezierPolyMat(x, y, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))    
    for k, (i, j) in enumerate(ij):         
        G[:, k] = ComputeBernsteinPoly(order, i, x) * ComputeBernsteinPoly(order, j, y)
    return G

def ConstructControlPoints(qz, order):
    ij = itertools.product(range(order+1), range(order+1))
    qPoints = np.zeros((qz.size, 3))
    step = 0.5 if order==2 else 1./3        
    for k, (i,j) in enumerate(ij):
        qPoints[k,:] = np.array([i*step, j*step, qz[k]])
    return qPoints

def GenerateSurfaceMesh(gridsize, controlsZ, u, v, xMin, yMin, xRange, yRange, order):
    nu, nv = gridsize, gridsize
    uu, vv = np.meshgrid(np.linspace(u.min(), u.max(), nu),
                         np.linspace(v.min(), v.max(), nv))

    GG = ComputeBezierPolyMat(uu.ravel(), vv.ravel(), order)    
    surfacePoints = np.dot(GG, controlsZ)
    
    # translate back to x, y positions
    zz = np.reshape(surfacePoints, uu.shape)
    xx = uu*xRange + xMin
    yy = vv*yRange + yMin
    return xx, yy, zz


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## main start
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#params
pUseMask = 1
pDrawControlPoints = 0
pFitPenalty = 0.01
#filter design
pGridsize = 500
pHeightThreshold = 2.0 #mm
pGradientThreshold = 0.001

# prototyping file
cloudDataFile = "C:/projects/cloudPoints.csv"
surfaceTargetClickFile = "C:/projects/surfaceTargClick.csv"
cloudData = np.genfromtxt(cloudDataFile, delimiter=',')
cloud = cloudData[:,0:3]
cloudWeights = cloudData[:,3]
cloudSurfLabels = cloudData[:,4]
surfaceData = np.genfromtxt(surfaceTargetClickFile, delimiter=',')
surfacePos = surfaceData[0,:]
surfaceTargDir = -1.0 * surfaceData[1,:] # normal vector is opposite of targetting

cloud = cloud[cloudSurfLabels == 1.0]
cloudWeights = cloudWeights[cloudSurfLabels == 1.0]

order = 3  # order of polynomial
x, y, z = cloud.T
#SX, SY, SZ = cloud[cloudSurfLabels == 1.0].T

W = cloudWeights * np.eye(x.size, x.size) + 0.01 * np.eye(x.size, x.size)


if x.size < 25:
    order = 2

# parametrize x y to u, v
xDataMin = x.min()
yDataMin = y.min()
xDataRange = x.max() - x.min()
yDataRange = y.max() - y.min()
u = (x - xDataMin) / xDataRange
v = (y - yDataMin) / yDataRange

# make Matrix:
bPolyMat = ComputeBezierPolyMat(u, v, order)

# weighted version, q = (X'WX)^-1 * X'Wz
# penalized version (regularized least squares) q = (X'X + vI)^-1 * X'z 
# weighted penalized version q = (X'WX + vI)^-1 * X'Wz

theta = pFitPenalty * np.eye((order + 1)**2)
innerMat = np.linalg.multi_dot([bPolyMat.T, W, bPolyMat]) + theta
conditionNumber = np.linalg.cond(innerMat)    
print("condition: " + str(conditionNumber))

# scipy notes
# inv vs pinv vs pinv2 
# inv uses lapack getrf, getri, getri_lwork
# pinv uses scipy.lstsq which uses lapack gelss, gelsd depending on matrix 
#   which inevitably uses SVD too
# pinv2 uses lapack gesdd, gesvd, gesXd ? depending..
# by construction RLS innerMat is symmetric and pos def so .inv is fine 
q = np.dot(scipy.linalg.inv(innerMat), np.linalg.multi_dot([bPolyMat.T, W, z]))

# this is just for visual
controlPoints = ConstructControlPoints(q, order)
print(controlPoints)

# Evaluate it on a grid...
startIt = time.time()
xx, yy, zz = GenerateSurfaceMesh(pGridsize, controlPoints[:,2], u, v, xDataMin, yDataMin, xDataRange, yDataRange, order)


##############
# image filter idea, should pull this to separate class & file
sobX = ndimage.sobel(zz, axis=1)
sobY = ndimage.sobel(zz, axis=0)
sobG = np.hypot(sobX, sobY)
sobXX = ndimage.sobel(sobX, axis=1)
sobYY = ndimage.sobel(sobY, axis=0)
sobXY = ndimage.sobel(sobX, axis=0)
hessDet = sobXX * sobYY - sobXY * sobXY # note element-wise multiplication

# remove extra pad from computation (maybe need more)
cut = 2    
sobG = sobG[cut:-cut, cut:-cut]
sobXX = sobXX[cut:-cut, cut:-cut]
hessDet = hessDet[cut:-cut, cut:-cut]


# filter test
maskedG = np.zeros(sobG.shape)
zDepth = zz.max() - zz.min()
print('z depth: ' + str(zDepth))
newTarget = None

# if not a simply flat surface, check gradients
if zDepth > pHeightThreshold:  
    
    maskedG[sobG < pGradientThreshold] = 1.0
    hitCount = maskedG.sum()
    print('hits: ' + str(hitCount))      
    
    # check hits for laplacian        
    if hitCount > 0:            

        hits = np.argwhere(maskedG == 1)            
        localMax = -1
        localUV = None
        for hit in hits:

            thisHD = hessDet[hit[0], hit[1]]
            thisSxx = sobXX[hit[0], hit[1]]
            zVal = zz[hit[0] + cut, hit[1] + cut]
            if thisHD > 0 and thisSxx < 0 and zVal > localMax:
                localMax = zVal
                localUV = hit

        if localUV is not None:
            newTargX = xx[localUV[0] + cut, localUV[1] + cut]
            newTargY = yy[localUV[0] + cut, localUV[1] + cut]
            newTarget = (newTargX, newTargY, localMax)


print("--new targ--")
print(newTarget)
endIt = time.time()
print('compute time: ' + str(endIt - startIt))
#############        


# control points from uvz to xyz (just for graphing)
xc, yc, zc = controlPoints.T
xc = xc*xDataRange + xDataMin
yc = yc*yDataRange + yDataMin


# downsample mesh for 3d plotting, or this case just generate a sparser one
vizGrid = 30
xx, yy, zz = GenerateSurfaceMesh(vizGrid, controlPoints[:,2], u, v, xDataMin, yDataMin, xDataRange, yDataRange, order)

# create mask for viewing purposes
if pUseMask == 1:
    z_inf = z.min()-1
    z_sup = z.max()+1
    zz[zz<z_inf] = z_inf
    zz[zz>z_sup] = z_sup 

# Plotting        
fg = plt.figure()  #figsize=plt.figaspect(1.5))

# image
axIm = fg.add_subplot(2,3,1)
axIm.imshow(zz, cmap=plt.cm.gray)    

# other images
ax2 = fg.add_subplot(2,3,2)
ax3 = fg.add_subplot(2,3,3)
ax4 = fg.add_subplot(2,3,4)
ax5 = fg.add_subplot(2,3,5)

binSobXX = np.zeros(sobXX.shape)
binSobXX[sobXX < 0] = 1.0

binHessD = np.zeros(hessDet.shape)
binHessD[hessDet > 0] = 1.0
binHessD[sobXX >= 0] = 0.0

ax2.imshow(scipy.misc.imresize(sobG, (30,30)), cmap=plt.cm.hot)
ax3.imshow(maskedG, cmap=plt.cm.gray)
ax4.imshow(scipy.misc.imresize(binSobXX, (30,30)), cmap=plt.cm.gray)
ax5.imshow(scipy.misc.imresize(binHessD, (30,30)), cmap=plt.cm.gray)



# surface
#ax = fg.add_subplot(2,3,6, projection='3d')
fg2 = plt.figure()
ax = fg2.add_subplot(1,1,1, projection='3d')
ls = LightSource(270, 45)
rgb = ls.shade(zz, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

ax.plot3D(x, y, z, "o", color="blue")
#ax.plot3D(SX, SY, SZ, "^", color="red")


# control points
if pDrawControlPoints == 1:
    ax.plot3D(xc, yc, zc, "^", color="green")
# draw new target
if newTarget is not None:
    ntx, nty, ntz = newTarget
    ax.plot3D([ntx], [nty], [ntz], "^", color="red")        
# add surface ray
rayLen = round(zz.max() - zz.min()) * 0.25
ax.quiver(surfacePos[0], surfacePos[1], surfacePos[2],
    surfaceTargDir[0], surfaceTargDir[1], surfaceTargDir[2],
    length=rayLen, color='r')

#fg.canvas.draw()
fg2.canvas.draw()
plt.show()
