from Multivariate_LeaniarRegression_Model import Multivariate_LeaniarRegression_Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
uurl=r"C:\Users\Hadeel\Desktop\4th year\4th year 1st term\Machine_Learning\Task_2\univariateData.dat"
unidata =np.loadtxt(uurl,delimiter=',')
X = unidata[:,0]
y = unidata[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=5)
X_train=np.array([X_train])
X_test=np.array([X_test])
Y_train=np.array([(Y_train)])
Y_test=np.array([(Y_test)])
#print(Y_train.shape , X_train.shape, X_test.shape,Y_test.shape)
X_train=(X_train.T)
Y_train=(Y_train.T)
X_test=(X_test.T)
Y_test=(Y_test.T)
#print(Y_train.shape , X_train.shape, X_test.shape,Y_test.shape)
model = Multivariate_LeaniarRegression_Model(X_train.shape[1])
model.fit(X_train,Y_train,1500, lr = 0.02)
pred=[]
for i in range(0,20):
    pred.append(model.predict(X_test[i]))

umse=model.EvaluatePerformance(pred,Y_test)
print('Mean square error',umse)
plt.figure(2)
plt.plot(X_test, pred, "r-")
plt.plot(X_test, Y_test, "b.")
plt.show()
## there are  2 plots : the first is for costs per iteration and the second shows the result plot 
## after closing the first figure the second will appear