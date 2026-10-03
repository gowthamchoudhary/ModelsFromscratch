import numpy as np

class SVM_POLY_KERNEL:
    def __init__(self,learning_rate,epochs,c,deg,gamma):
        self.learning_rate = 0.01
        self.epochs = 1000
        self.weights = None
        self.alpha = None
        self.c  = 1.0
        self.deg = deg
        self.b=None
        self.n_features = None
        self.K = None
        self.gamma = gamma
    def kernel_cals(self,X_train):
        K = np.zeros((self.n_features,self.n_features))
        for i in range(self.n_features):
            for j in range(self.n_features):
                K[i,j] = (self.gamma*np.dot(X_train[i].T,X_train[j])+self.c)**self.deg
        return K
    def build_Q(self,y_train):
        return np.dot(y_train.T,y_train)@self.K

    def fit(self,X_train,y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.n_features = X_train.shape[1]
        self.K = self.kernel_cals(X_train)
        self.alpha = np.zeros(self.n_features)
        self.b  = 0.0
    def cal_loss(self,y_train):
        err = []
        for i in range(self.n_features):
            for j in range(self.n_features):
                f = self.alpha[j]*y_train[j]*self.K[i,j]+self.b
                e = f-y_train[i]

