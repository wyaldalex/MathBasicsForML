import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('Python: {}'.format(sys.version))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))

#generate 2D meshgrid
nx, ny = (100,100)
x = np.linspace(0,10,nx)
y = np.linspace(0,10,ny)
xv , yv = np.meshgrid(x,y)


#define a plotting function
def f(x,y):
    return x * (y**2)

# calculate Z for each x,y point
z = f(xv,yv)
print(z.shape)

#make a color plot to display the data
plt.figure(figsize=(14,12))
plt.pcolor(xv,yv,z)
plt.title('2D Color plot of f(x,y) = xy^2')
plt.colorbar
#plt.show()


#generate 2D meshgrid for the gradient
nx, ny = (10,10)
x = np.linspace(0,10,nx)
y = np.linspace(0,10,ny)
xg , yg = np.meshgrid(x,y)

#calculate gradient of f(x,y) Notice the inverted y and x of Gy and Gx, as np does rows first
Gy, Gx = np.gradient(f(xg,yg))

#make a color plot to display the data
#The lenght of the arrow/vector will indicate the rate(slope) of change at the coordinates
plt.figure(figsize=(14,12))
plt.pcolor(xv,yv,z)
plt.title('Gradient of f(x,y) = xy^2')
plt.colorbar
plt.quiver(xg,yg,Gx,Gy, scale = 1000, color = 'w')
#plt.show()

#verify by calculating the partial derivatives
def ddx(x,y):
    return y ** 2

def ddy(x,y):
    return 2 * x * y

Gx = ddx(xg,yg)
Gy = ddy(xg,yg)

plt.figure(figsize=(14,12))
plt.pcolor(xv,yv,z)
plt.title('Plot of the partial derivatives [y^2, 2xy]')
plt.colorbar
plt.quiver(xg,yg,Gx,Gy, scale = 1000, color = 'w')
plt.show()

