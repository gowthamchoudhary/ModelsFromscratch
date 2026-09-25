import numpy as np

class SVM_SOFT:
    def __init__(self,learning_rate,C,epochs):
        self.learning_rate = learning_rate
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = None

    # def hinge_loss(self,x,y):
    #     reg = 0.5*(self.w**2)
    #     for i in range(x.shape[0]):
    #         opt_term = y[i]*(self.w@x+self.b)
    #         loss = reg+self.c*(max(0,1-opt_term))
    #     return loss[0][0]
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b=0
        for i in range(self.epochs):
            scores = X @ self.w + self.b
            margin = y*scores
            violating = margin<1
            dw = self.w.copy()
            db = 0.0
            dw -= (
                self.C / n_samples
            ) * np.sum(
                y[violating, None] * X[violating],
                axis=0
            )

            db = -(
                self.C / n_samples
            ) * np.sum(
                y[violating]
            )
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):

        scores = X @ self.w + self.b

        return np.where(scores >= 0, 1, -1) 
X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [1, 1],
    [2, 1],
    [3, 2]
])

y = np.array([
    1,
    1,
    1,
    -1,
    -1,
    -1
])


model = SVM_SOFT(
    learning_rate=0.001,
    epochs=5000,
    C=1
)

model.fit(X, y)

predictions = model.predict(X)

print("w:", model.w)
print("b:", model.b)
print("Predictions:", predictions)
print("Actual:", y)

accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)