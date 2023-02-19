import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

arrayA = np.arange(9) - 3
matrixB = arrayA.reshape((3,3))
print(matrixB)

print("\n--------------------------Euclidian L2 Norm---------------------------------")
#Euclidian (L2) Norm - default
print(np.linalg.norm(arrayA))
print(np.linalg.norm(matrixB))

print("\n--------------------------Frogenius Norm---------------------------------")
#The Frogenius norm is the L2 norm for a matrix
print(np.linalg.norm(matrixB,'fro'))

print("\n--------------------------The Max Norm---------------------------------")
#The max norm(P == infinity)
print(np.linalg.norm(arrayA,np.inf))
print(np.linalg.norm(matrixB,np.inf))

print("\n--------------------------Vector Normalization --------------------------------")
#Normationalization to produce unit vector
norm = np.linalg.norm(arrayA)
A_unit = arrayA / norm
print(A_unit)

#Magniute of a unit vector is equal to 1
print(np.linalg.norm(A_unit)) #close to 1

print("\n--------------------------Eigenvalues Decomposition--------------------------------")
A = np.diag(np.arange(1,4))
print(A)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues: " , eigenvalues)
print("Eigenvectors: \n" , eigenvectors)


print("\n--------------------------Verify Eigendecomposition--------------------------------")
matrixB = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors,matrixB).astype(int)
print("We should get our original matrix \n",output)
print("Original matrix \n", A)

print("\n--------------------------Plot Eigenvectors--------------------------------")
origin = [0,0,0]

fig = plt.figure(figsize=(18,10))
fig.suptitle("Effects of Eigenvalues and Eigenvectors, no angle changes only scale")
ax1 = fig.add_subplot(121, projection = '3d')
ax1.quiver(origin,origin,origin,
           eigenvectors[0, :],
           eigenvectors[1, :],
           eigenvectors[2, :],
           color = 'k')
ax1.set_xlim([-3,3])
ax1.set_ylim([-3,3])
ax1.set_zlim([-3,3])
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.view_init(15,30)
ax1.set_title('Before Multiplication')


new_eig = np.matmul(A, eigenvectors)
#fig2 = plt.figure(figsize=(18,10))
ax2 = fig.add_subplot(122, projection = '3d')
ax2.quiver(origin,origin,origin,
           new_eig[0, :],
           new_eig[1, :],
           new_eig[2, :],
           color = 'k')
ax2.plot((eigenvalues[0]*eigenvectors[0]), (eigenvalues[1]*eigenvectors[1]),
         (eigenvalues[2]*eigenvectors[2]), 'rX')
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])
ax2.set_zlim([-3,3])
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.view_init(15,30)
ax2.set_title('After Multiplication')

plt.show()

















