import numpy as np
import matplotlib.pyplot as plt

class Multivariate_LeaniarRegression_Model:
    def __init__(self,n_f):
        #initialization
        self.w = np.random.randn(1,n_f)
        self.b = 0
    def computeCost (self,y_pred,y):
        m = y.shape[0]
        cost = ((y_pred - y)**2)/(2*m)
        return np.sum(cost)
    def gradientDescent(self,X,y):
        m = y.shape[0]
        y_hat = self.predict(X)
        dw = np.dot((y_hat - y).T , X)*(1/m)
        db = np.sum(y_hat - y)*(1/m) 
        return dw, db
    def fit(self,X,y,n_iteration,lr):
        self.w_history = self.w
        self.b_history = [self.b]
        costArr = []
        y_pred = self.predict(X)
        initial_cost = self.computeCost(y_pred,y)
        costArr.append(initial_cost)
        for i in range(1,n_iteration+1):
            dw,db = self.gradientDescent(X,y)
            self.w = self.w - lr*dw
            self.b = self.b - lr*db
            y_pred = self.predict(X)
            cost = self.computeCost(y_pred, y)
            costArr.append(cost)
            self.w_history = np.append(self.w_history, self.w, axis = 0)
            self.b_history.append(self.b)  
        print('learning Rate:' ,lr)
        print('number of iterations:', n_iteration)
        print('Final cost value: ',cost)
        plt.plot(costArr,'-o')
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
    def predict(self,X):
        y_pred = np.dot(X, self.w.T)+self.b 
        return y_pred
    def EvaluatePerformance(self,y_pred,y):
        mse=np.mean((y - y_pred)**2)
        return mse
