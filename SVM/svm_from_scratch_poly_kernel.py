import numpy as np

class SVM_POLY_KERNEL:
    def __init__(self,learning_rate,epochs,c,tol,deg,gamma):
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
        self.tol = tol
    def kernel_cals(self,X_train,n_samples):
        K = np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = (self.gamma*np.dot(X_train[i].T,X_train[j])+self.c)**self.deg
        return K
    def predict_score(self,X_train,y_train):
        f = []
        for i in range(X_train):
            for j in range(self,X_train):
                f.append( self.alpha[j]*y_train[j]*self.K[i,j]+self.b)
        return f
    def calculate_error(self,f,y_train):
        err = []
        for i in range(f.len()):
            err.append(f[i]-y_train[i])
        return err
    def check_kkt(self,y_train,n_samples,err):
        violations = []
        for i in range(n_samples):
            violations = (y_train[i]*err[i]<-self.tol and self.alpha[i]<self.c)
        return violations
    def calculate_all_errors(self,X_train,y_train,n_samples):
        f = self.predict_score(X_train,y_train)
        err = self.calculate_error(f,y_train)
        return self.check_kkt(y_train,n_samples,err)
    def select_j(self, err, n_samples):
        selected_j = []
        for i in range(n_samples):
            max_diff = -1
            best_j = None
            for j in range(n_samples):
                diff = np.abs(err[i] - err[j])
                if diff > max_diff:
                    max_diff = diff
                    best_j = j
            selected_j.append(best_j)
        return selected_j        
    def calculate_H_L(self,n_samples,y_train,selected_j):
        for i in range(n_samples):
            for j in range(n_samples):
                if y_train[i]!=y_train[j]:
                    l = np.argmax(0,self.alpha[j]-self.alpha[i])
                    H  = np.argmin(self.c,self.c+self.alpha[j]-self.alpha[i])
                else:
                    l = np.argmax(0,self.alpha[i]+self.alpha[j]-self.c)
                    H = np.argmin(self.c,self.alpha[i]+self.alpha[j])
                    
                    
            

    def build_Q(self,y_train):
        return np.dot(y_train.T,y_train)@self.K

    def fit(self,X_train,y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        n_samples,self.n_features = X_train.shape
        self.K = self.kernel_cals(X_train,n_samples)
        self.alpha = np.zeros(n_samples)
        self.b  = 0.0
    def cal_loss(self,y_train):
        err = []
        for i in range(self.n_features):
            for j in range(self.n_features):
                f = self.alpha[j]*y_train[j]*self.K[i,j]+self.b
                e = f-y_train[i]

