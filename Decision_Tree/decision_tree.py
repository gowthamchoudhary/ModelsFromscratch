import numpy as np


class Node:
    def __init__(self,feature=None,threshold=None,left_node=None,right_node=None,prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.prediction = prediction
class Decision_Tree_Classifier:
    def __init__(self,max_depth=3):
        self.classes = []
        self.max_depth = max_depth
        self.root = None
    def cal_entropy(self,class_prob:list):
        class_prob = np.asarray(class_prob)
        return np.sum([-p*np.log2(p) for p in class_prob if p>0])
    def Gini(self,class_prob):
        class_prob = np.asarray(class_prob)
        return 1-np.sum(class_prob**2)
    def weighted_gini(self,left_y,right_y):
        gini_prob_left = self.Gini(self.class_prob(left_y))
        gini_prob_right = self.Gini(self.class_prob(right_y))
        l  = len(left_y)+len(right_y)
        return ((len(left_y)/l)*gini_prob_left + (len(right_y)/l)*gini_prob_right)
    def cal_Threshold(self,x_feature):
        x_feature = np.asarray(x_feature)
        x_feature.sort()
        mp = [(x_feature[j]+x_feature[j+1])/2 for j in range(len(x_feature)-1)]
        return mp

    def class_prob(self,y):
        y = np.array(y)
        unq = np.unique(y)
        class_probabilities = []
        for i in unq:
            class_probabilities.append((y==i).sum()/len(y))
        return class_probabilities
        
    def best_split(self,x_feature,y):
        threshold_candidates = self.cal_Threshold(x_feature)
        final_gini_scores=[]
        for i in threshold_candidates:
            left = y[x_feature<i]
            right = y[x_feature>=i]
       
            w_g = self.weighted_gini(left,right)
            final_gini_scores.append(w_g)
        best_index = np.argmin(final_gini_scores)
        best_threshold = threshold_candidates[best_index]
        best_gini = final_gini_scores[best_index]
        return best_threshold , best_gini
    



