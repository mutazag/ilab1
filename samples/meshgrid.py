#%% 
import numpy as np 
import matplotlib.pyplot as plt


#%%
u = np.linspace(-2,2,65)
v = np.linspace(-1,1,33)

#%%
X,Y = np.meshgrid(u,v)

#%%
Z = X**2 /25 + Y**2/4
# Z = np.sin(X) + np.cos(Y)

#%%
plt.pcolor(Z)
plt.colorbar()
plt.show()

#%%
plt.pcolor(X, Y, Z,  cmap='gray') # show the meshgrid to label
plt.colorbar()
plt.axis('tight')
plt.show()

#%%
a = [1,2,3,4]
b = [10,11,12]
A,B = np.meshgrid(a,b)

#%% 
plt.contour(X, Y, Z, 30, cmap='autumn') # show the meshgrid to label
plt.colorbar()
plt.axis('tight')
plt.show()

#%%
plt.contourf(X, Y, Z, 30) # show the meshgrid to label
plt.colorbar()
plt.axis('tight')
plt.show()

#%%
plt.subplot(2,2,1)
plt.pcolor(Z)
plt.colorbar()

plt.subplot(2,2,2)
plt.pcolor(X, Y, Z,  cmap='gray') # show the meshgrid to label
plt.colorbar()

plt.subplot(2,2,3)
plt.contour(X, Y, Z, 30, cmap='autumn') # show the meshgrid to label
plt.colorbar()

plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 30) # show the meshgrid to label
plt.colorbar()

# plt.axis('tight')
plt.tight_layout()
plt.show()

#%%
