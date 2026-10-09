# import numpy as np

# class SVM_POLY_KERNEL:
#     def __init__(self,learning_rate,epochs,c,tol,deg,gamma):
#         self.learning_rate = 0.01
#         self.epochs = 1000
#         self.weights = None
#         self.alpha = None
#         self.c  = 1.0
#         self.deg = deg
#         self.b=None
#         self.n_features = None
#         self.K = None
#         self.gamma = gamma
#         self.sup_vec = []
#         self.tol = tol
#     def kernel_cals(self,X_train,n_samples):
#         K = np.zeros((n_samples,n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 K[i,j] = (self.gamma*np.dot(X_train[i].T,X_train[j])+self.c)**self.deg
#         return K
#     def predict_score(self,X_train,y_train):
#         f = []
#         for i in range(X_train):
#             for j in range(self,X_train):
#                 f.append( self.alpha[j]*y_train[j]*self.K[i,j]+self.b)
#         return f
#     def calculate_error(self,f,y_train):
#         err = []
#         for i in range(f.len()):
#             err.append(f[i]-y_train[i])
#         return err
#     def check_kkt(self,y_train,n_samples,err):
#         violations = []
#         for i in range(n_samples):
#             violations = (y_train[i]*err[i]<-self.tol and self.alpha[i]<self.c)
#         return violations
#     def calculate_all_errors(self,X_train,y_train,n_samples):
#         f = self.predict_score(X_train,y_train)
#         err = self.calculate_error(f,y_train)
#         return self.check_kkt(y_train,n_samples,err)
#     def select_j(self, err, n_samples):
#         selected_j = []
#         for i in range(n_samples):
#             max_diff = -1
#             best_j = None
#             for j in range(n_samples):
#                 diff = np.abs(err[i] - err[j])
#                 if diff > max_diff:
#                     max_diff = diff
#                     best_j = j
#             selected_j.append(best_j)
#         return selected_j        
#     def calculate_H_L(self,alpha_i,alpha_j,y_i,y_j):
#         if y_i == y_j:
#             l = np.argmax(0,alpha_j+alpha_i-self.c)
#             H = np.argmin(0,alpha_i+alpha_j)
#         else:
#             l = np.argmax(0,alpha_j-alpha_j)
#             H = np.argmin(self.c,self.c+alpha_i+alpha_j)
#         return l,H
#     def calculate_eta(self,i,j):

#         return self.K[i,i]+self.K[j,j]-2*self.K[i,j]
#     def update_alpha_j(self,i,j,err_i,err_j,y_j,alpha_j,l,H):
#         n = self.calculate_eta(i,j)
#         if n<=0:
#             return alpha_j
#         new_alpha_J = alpha_j+(y_j*(err_i-err_j))/n
#         #condition
#         if alpha_j<l and alpha_j>=H:
#             # self.alpha[j]= new_alpha_J
#             return new_alpha_J
#     def update_alpha_i(self,i,j,y_train,alpha_j_new,alpha_j_old,alpha_i):
#         alpha_i_new = alpha_i +y_train[i]*y_train[j]*(alpha_j_old-alpha_j_new)
#         return alpha_i_new
#     def calculate_b(self,b_old,
#                         E_i,
#                         E_j,
#                         alpha_i_old,
#                         alpha_i_new,
#                         alpha_j_old,
#                         alpha_j_new,
#                         y_i,
#                         y_j,
#                         K_ii,
#                         K_ij,
#                         K_jj,
#                         C):
#         b1 = b_old-E_i-y_i*(alpha_i_new-alpha_i_old)*K_ii-y_j*(alpha_j_new-alpha_j_old)*K_ij
#         b2 = b_old -E_j-y_i*(alpha_i_new-alpha_i_old)*K_ij - y_j*(alpha_j_new-alpha_j_old)*K_jj
#         if alpha_i_new<C and 0<alpha_i_new:
#             return b1
#         elif alpha_j_new<C and  alpha_i_new:
#             return b2
#         else :
#             return (b1+b2)/2        
     
#     def build_Q(self,y_train):
#         return np.dot(y_train.T,y_train)@self.K
#     def calculate_sv(self,threshold):
#         support_vectors = []
#         for i in self.alpha:
#             if i >= threshold:
#                 support_vectors.append(i)
#         self.sup_vec = support_vectors
#         return support_vectors
        
#     def decision_function(self,x_test,y_test):
#         for i in range(self.sup_vec.len()):
#             f_x = np.sum(self.alpha[i]*y_test[i]*self.K[x_test[i],x_test]+self.b)
#         return f_x
#     def predict(self,f_x):
#         positive,negative = np.where(f_x>=0)
#         return positive,negative

#     def fit(self,X_train,y_train):
#         X_train = np.asarray(X_train)
#         y_train = np.asarray(y_train)
        
#         n_samples,self.n_features = X_train.shape
#         self.K = self.kernel_cals(X_train,n_samples)
#         self.alpha = np.zeros(n_samples)
#         label = []
#         for i in y_train:
#             if i not in label:
#                 label.append(i)
                

#         self.b  = 0.0
#         err = self.calculate_error(y_train)
#         violations = self.check_kkt(y_train,n_samples,err)
#         for i in range(n_samples):
#             fi = self.predict_score(X_train,y_train)
#             err



import numpy as np


class SVM_POLY_KERNEL:

    def __init__(
        self,
        C=1.0,
        tol=1e-3,
        deg=2,
        gamma=1.0,
        coef0=1.0,
        max_passes=10,
        max_iter=10000
    ):
        self.C = C
        self.tol = tol
        self.deg = deg
        self.gamma = gamma
        self.coef0 = coef0
        self.max_passes = max_passes
        self.max_iter = max_iter

        self.alpha = None
        self.b = 0.0
        self.K = None
        self.X_train = None
        self.y_train = None

        self.sup_vec = None
        self.sup_y = None
        self.sup_alpha = None

    def kernel_cals(self, X1, X2):
        return (
            self.gamma * (X1 @ X2.T) + self.coef0
        ) ** self.deg

    def predict_score(self, X):
        K_test = self.kernel_cals(X, self.X_train)

        return K_test @ (self.alpha * self.y_train) + self.b

    def calculate_error(self, scores):
        return scores - self.y_train

    def check_kkt(self, i, error):
        yiE = self.y_train[i] * error

        return (
            (yiE < -self.tol and self.alpha[i] < self.C)
            or
            (yiE > self.tol and self.alpha[i] > 0)
        )

    def select_j(self, i, errors):
        candidates = np.arange(len(self.y_train))
        candidates = candidates[candidates != i]

        if len(candidates) == 0:
            return None

        differences = np.abs(errors[i] - errors[candidates])
        return candidates[np.argmax(differences)]

    def calculate_H_L(self, i, j):
        ai = self.alpha[i]
        aj = self.alpha[j]
        yi = self.y_train[i]
        yj = self.y_train[j]

        if yi != yj:
            L = max(0.0, aj - ai)
            H = min(self.C, self.C + aj - ai)
        else:
            L = max(0.0, ai + aj - self.C)
            H = min(self.C, ai + aj)

        return L, H

    def calculate_eta(self, i, j):
        return (
            self.K[i, i]
            + self.K[j, j]
            - 2 * self.K[i, j]
        )

    def update_alpha_j(self, i, j, errors, L, H):
        eta = self.calculate_eta(i, j)

        if eta <= 0:
            return None

        aj_old = self.alpha[j]
        yj = self.y_train[j]

        aj_new = aj_old + (
            yj * (errors[i] - errors[j])
        ) / eta

        aj_new = np.clip(aj_new, L, H)

        if abs(aj_new - aj_old) < 1e-5:
            return None

        return aj_new

    def update_alpha_i(self, i, j, aj_old, aj_new):
        ai_old = self.alpha[i]
        yi = self.y_train[i]
        yj = self.y_train[j]

        ai_new = ai_old + yi * yj * (aj_old - aj_new)
        return ai_new

    def calculate_b(
        self, i, j, errors, ai_old, ai_new, aj_old, aj_new
    ):
        yi = self.y_train[i]
        yj = self.y_train[j]

        b_old = self.b

        b1 = (
            b_old - errors[i]
            - yi * (ai_new - ai_old) * self.K[i, i]
            - yj * (aj_new - aj_old) * self.K[i, j]
        )

        b2 = (
            b_old - errors[j]
            - yi * (ai_new - ai_old) * self.K[i, j]
            - yj * (aj_new - aj_old) * self.K[j, j]
        )

        if 0 < ai_new < self.C:
            return b1
        elif 0 < aj_new < self.C:
            return b2
        else:
            return (b1 + b2) / 2

    def calculate_sv(self, threshold=1e-5):
        mask = self.alpha > threshold

        self.sup_vec = self.X_train[mask]
        self.sup_y = self.y_train[mask]
        self.sup_alpha = self.alpha[mask]

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train).ravel()

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train sizes must match.")

        classes = np.unique(y_train)
        if len(classes) != 2:
            raise ValueError("This SVM supports binary classification only.")

        self.classes_ = classes
        self.X_train = X_train
        self.y_train = np.where(y_train == classes[0], -1.0, 1.0)

        n_samples = len(X_train)

        self.K = self.kernel_cals(X_train, X_train)
        self.alpha = np.zeros(n_samples, dtype=float)
        self.b = 0.0

        passes = 0
        iterations = 0

        while passes < self.max_passes and iterations < self.max_iter:
            num_changed = 0

            scores = self.predict_score(self.X_train)
            errors = self.calculate_error(scores)

            for i in range(n_samples):
                if not self.check_kkt(i, errors[i]):
                    continue

                j = self.select_j(i, errors)
                if j is None:
                    continue

                ai_old = self.alpha[i]
                aj_old = self.alpha[j]

                L, H = self.calculate_H_L(i, j)

                if L == H:
                    continue

                aj_new = self.update_alpha_j(i, j, errors, L, H)
                if aj_new is None:
                    continue

                ai_new = self.update_alpha_i(i, j, aj_old, aj_new)

                self.alpha[i] = ai_new
                self.alpha[j] = aj_new

                self.b = self.calculate_b(
                    i, j, errors,
                    ai_old, ai_new,
                    aj_old, aj_new
                )

                num_changed += 1

                
                scores = self.predict_score(self.X_train)
                errors = self.calculate_error(scores)

            iterations += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        self.calculate_sv()
        return self

    def decision_function(self, X_test):
        X_test = np.asarray(X_test, dtype=float)

        K_test = self.kernel_cals(X_test, self.sup_vec)

        return K_test @ (self.sup_alpha * self.sup_y) + self.b

    def predict(self, X_test):
        scores = self.decision_function(X_test)

        return np.where(
            scores >= 0,
            self.classes_[1],
            self.classes_[0]
        )
X = np.array([
    [1, 1],  # A
    [2, 1],  # B
    [1, 3],  # C
    [2, 3]   # D
])

y = np.array([1, 1, -1, -1])

model = SVM_POLY_KERNEL(
    C=1.0,
    tol=1e-3,
    deg=2,
    gamma=1.0,
    coef0=1.0
)

model.fit(X, y)

print("Kernel matrix:\n", model.K)
print("Alphas:", model.alpha)
print("Bias:", model.b)
print("Support vectors:\n", model.sup_vec)
print("Decision scores:", model.decision_function(X))
print("Predictions:", model.predict(X))
