import numpy as np

a = np.array([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,0,1,0],[0,0,0,1]])
b = np.zeros(a[0].shape).astype(int)
for i in range(0,a.shape[0]):
    a[i] -= b
    b += a[i]

a[a>0]=1
a[a<=0]=0

print(a)
