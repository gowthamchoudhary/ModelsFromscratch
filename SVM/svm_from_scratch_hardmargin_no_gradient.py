import numpy as np
from scipy.optimize import minimize


class SVM_hard_margin:
    def __init__(self,w=None,b=None ):
        self.w = w
        self.b=b
    def constraint(self,params,X,y):
        w = params[:-1]
        b=params[-1]
        scores = X @ w + b
        return y(scores)-1
    def objective(self,params):
        w  = params[:-1]
        return 0.5*np.dot(w,w)
    def fit(self,X,y):
        X=np.asarray(X)
        y=np.asarray(y)
        n_features = X.shape[1]
        initial_w = np.zeros(n_features)
        initial_b=0.0
        initial_param = np.concatenate([initial_w,[initial_b]])
        constraints={
            "type":"ineq",
            "fun":self.constraint,
            "args":(X,y)
        }
        result =minimize(
            fun=self.objective,
            x0=initial_param,
            args=(),
            constraints=constraints,
            method="SLSQP"
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed:{result.message}")
        self.w = result.x[:-1]
        self.b = result.x[-1]
        return self
    def predict(self,X):
        X = np.asarray(X)
        scores = X@self.w+self.b
        return np.where(scores>=0,1,-1)
    

        