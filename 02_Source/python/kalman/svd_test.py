import numpy as np

# Point Pairs 0: 8.757587->6.062546, -15.012836->-14.550013, -3.333965->-2.437946 centriod: (59.785099, -4.783874, -17.156046) -> (58.692993, -4.116979, -16.729830) 
# Point Pairs 229: -14.784103->-12.193405, 25.183039->25.203478, 8.315681->7.594459 
# Point Pairs 351: 6.026516->6.130852, -10.170202->-10.653465, -4.981716->-5.156517

Y = [[8.757587, -14.784103, 6.026516],[-15.012836,25.183039,-10.170202],[-3.333965,8.315681,-4.981716]]
X = [[6.062546, -12.193405, 6.130852],[-14.550013,25.203478, -10.653465],[-2.437946, 7.594459, -5.156517]]

S = np.matmul(X,np.transpose(Y))

U, Sig_diag, Vt = np.linalg.svd(S)

print("u"+ str(U))
print("Sig_diag"+ str(Sig_diag))
print("Vt"+ str(Vt))

Sig = np.diag(Sig_diag)
print("Sig ="+str(Sig))
Sdet = np.identity(3)
Sdet[2][2] = 
print(np.matmul(U,np.matmul(Sig,Vt)))
print("Vt ="+str(Vt))
V = np.transpose(Vt)
print("V ="+str(np.transpose(Vt)))
I = np.identity(3)
I[2,2] = np.linalg.det(np.matmul(V,np.transpose(U)))

Rt = np.matmul(V,np.matmul(I,np.transpose(U)))

R = np.transpose(Rt)
print("R ="+str(R))
print("Eigvl(R) = "+str(np.linalg.eig(R)))