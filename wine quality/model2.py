#Import the libraries
import numpy as np 
import pandas as pd
from numpy import log,dot,exp,shape
from sklearn.model_selection import train_test_split

#get the data
data=pd.read_csv('Wine_data.csv')
data2=data.copy()
data2['quality']=data2["quality"]>5
data2 = data2.groupby('quality')

[X1_train, X1_test, y1_train, y1_test] = train_test_split(data2.get_group(1),data2.get_group(1)['quality'], test_size=0.30, random_state=42,shuffle = True)
[x2_train, x2_test, y2_train, y2_test] = train_test_split(data2.get_group(0),data2.get_group(0)['quality'], test_size=0.30, random_state=42,shuffle = True)

# Concatinating the training data and testing data
X_train = pd.concat([X1_train,x2_train])
X_test = pd.concat([X1_test,x2_test])
Y_train = pd.concat([y1_train,y2_train])
Y_test = pd.concat([y1_test,y2_test])

X_tr,X_te,y_tr,y_te = X_train,X_test,Y_train,Y_test

y_tr=np.array(y_tr)
y_te=np.array(y_te)

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr.iloc[:,i] = (X_tr.iloc[:,i] - np.mean(X_tr.iloc[:,i]))/np.std(X_tr.iloc[:,i])
def accuracy_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return accuracy
class LogidticRegression:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.001,iter=400):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
standardize(X_tr)
standardize(X_te)
obj1 = LogidticRegression()
model= obj1.fit(X_tr,y_tr)
y_pred = obj1.predict(X_te)
print(y_pred)

f1_score_te = accuracy_score(y_te,y_pred)
print(f1_score_te)

#dump the model
import pickle
filename='final_model.pickle'
pickle.dump(model,open(filename, 'wb'))