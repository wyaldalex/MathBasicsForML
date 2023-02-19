import sys
import numpy as np

print('Python: {}'.format(sys.version))
print('Python: {}'.format(np.__version__))

print("\n--------------------------Basics---------------------------------")
x = np.array((1,2,3))
print('Vector dimensions: {}'.format(x.shape))

matrixA = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print('Matrix dimensions: {}'.format(matrixA.shape))
print('Matrix size: {}'.format(matrixA.size))

#quick matrix defitions
print("\n--------------------------Quick matrix definitions---------------------------------")
oneMatrix = np.ones((5,5))
print(oneMatrix)

zerosMatrix = np.zeros((3,3,3))
print(zerosMatrix)
print('Matrix dimensions: {}'.format(zerosMatrix.shape))
print('Matrix size: {}'.format(zerosMatrix.size))

#Indexing
print("\n--------------------------Indexing---------------------------------")
matrixB = np.ones((3,3,3))
print(matrixB)
#these are mutable strucutres
matrixB[0,0,0] = 100
print("After mutation")
print(matrixB)

#Multiple dimension changes
print("\n--------------------------Multiple dimension changes---------------------------------")
matrixB[:,:,0] = 5
print("After mutation")
print(matrixB)

#Matrix operations
print("\n--------------------------Matrix operations---------------------------------")
matrixC = np.matrix([[1,4],[3,4]])
matrixD = np.matrix([[2,5],[5,7]])
print(matrixC)

print("Operation mutation")
print(matrixC + matrixD)
print("Operation mutation")
print(matrixC * matrixD)

print("\n--------------------------TRANSPOSE---------------------------------")
#Matrix transponse
matrixE = np.array(range(9)).reshape(3,3)
print(matrixE)
print(matrixE.T)

print("\n--------------------------TENSORS---------------------------------")
tensorA = np.ones((3,3,3,3,3,3,3,3,3,3))
print(tensorA.size)
print(tensorA.shape)
