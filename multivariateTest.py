from Multivariate_LeaniarRegression_Model import Multivariate_LeaniarRegression_Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
murl=r"C:\Users\Hadeel\Desktop\4th year\4th year 1st term\Machine_Learning\Task_2\multivariateData.dat"
multidata =np.loadtxt(murl,delimiter=',')
X = multidata[:,0:2]
y = multidata[:,2]
sc = MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
#print(X.shape)
y = multidata[:,2]
y=np.array([y])
y=(y.T)
for i in range(0,10):
    y[i]=float(y[i]/100)
#print(y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=5)
#print(Y_train.shape , X_train.shape, X_test.shape,Y_test.shape)
mmodel = Multivariate_LeaniarRegression_Model(X_train.shape[1])
mmodel.fit(X_train,Y_train,8000, lr = 0.0059)
pred=[]
for i in range(0,10):
    p=float(mmodel.predict(X_test[i]))
    pred.append(p)
mmse=mmodel.EvaluatePerformance(pred,Y_test)
print('Mean square error',mmse)#i didn't understand why mmse is large although the line looks good
plt.plot(X_test[:,0], pred, "r-")
plt.plot(X_test[:,0], Y_test, "b.")
plt.show()
## there are  2 plots : the first is for costs per iteration and the second shows the result plot 
## after closing the first figure the second will appear
