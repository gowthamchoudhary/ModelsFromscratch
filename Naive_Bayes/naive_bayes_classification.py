import numpy as np


class Naive_Bayes:
    def __init__(self):
       
        self.vocabulary = set()
        self.classes = set()
        self.class_count = {}
        self.prior_probability = {}
        self.word_counts = {}
        self.total_words_in_class = {}
        self.total_documents = 0
    def tokenizer(self,text):
        return text.lower().split()
    def get_classes(self,y_train):
        classes = set()
        for label in y_train:
            classes.add(label)
        return classes

    def create_vocubalary(self,x_train):
        vocabulary = set()
        for text in x_train:
            words = self.tokenizer(text)
            for word in words:
               vocabulary.add(word)
        return vocabulary
    def count_documents(self,y_train):
        class_count = {}
        for label in y_train:
            if label not in class_count:
                class_count[label]=0
            class_count[label]+=1
        return class_count
    def count_Words(self,x_train,y_train):
        words_count = {}
        total_words = {}
        for label in self.classes:
            words_count[label]={}
            total_words[label]=0
        for label,text in zip(y_train,x_train):
            words = self.tokenizer(text)
            for word in words:
                if word not in words_count[label]:
                    words_count[label][word] = 0
                words_count[label][word]+=1
                total_words[label]+=1
        return words_count,total_words
                        

    def calculate_Prior_prob(self):
        prior_prob = {}
        for label in self.class_count:
            prior_prob[label] = (self.class_count[label]/sum(self.class_count.values()))
        return prior_prob
    
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = self.get_classes(y_train)
        self.vocabulary  = self.create_vocubalary(x_train)
        self.class_count = self.count_documents(y_train)
        self.word_counts,self.total_words_in_class = self.count_Words(x_train,y_train)
        self.prior_probability = self.calculate_Prior_prob()
    def word_probability(self,word,label):
        word_count = self.word_counts[label].get(word,0)
        tot_words_count = self.total_words_in_class[label]
        len_voc = len(self.vocabulary)
        return ((word_count+1)/(tot_words_count+len_voc))
    def class_score(self,words,label):
        score = self.prior_probability[label]
        for word in words:
            probability = self.word_probability(word,label)
            score*=probability
        return score
    def predict(self,X_test):
        predictions = []
        for text in X_test:
            words = self.tokenizer(text)
            score={}
            for label in self.classes:
                score[label] = self.class_score(
                    words,label
                )
            prediction = max(score,key=score.get)
            predictions.append(prediction)
        return predictions
X_train = [
    "iron man is an amazing movie",
    "captain america is a great hero",
    "thor is funny and entertaining",
    "avengers is an excellent movie",
    "guardians of the galaxy is fantastic",
    "spider man is a wonderful superhero",
    "black panther is powerful and inspiring",
    "doctor strange is visually amazing",
    "thor ragnarok is hilarious and exciting",
    "avengers endgame is emotional and fantastic",
    "iron man has a great story",
    "captain america has an excellent story",
    "guardians of the galaxy has funny characters",
    "black panther has a powerful story",
    "spider man has amazing action",
    "doctor strange has beautiful visuals",
    "avengers has fantastic action scenes",
    "thor is a wonderful character",

    "iron man is a boring movie",
    "thor is a disappointing movie",
    "captain america is slow and boring",
    "avengers is a terrible movie",
    "guardians of the galaxy is boring",
    "spider man has a weak story",
    "black panther is disappointing",
    "doctor strange has a boring story",
    "thor ragnarok is disappointing",
    "avengers endgame is too long and boring",
    "iron man has a weak story",
    "captain america has a terrible story",
    "guardians of the galaxy has boring characters",
    "black panther has a weak story",
    "spider man has disappointing action",
    "doctor strange has terrible visuals",
    "avengers has boring action scenes",
    "thor is a disappointing character"
]

y_train = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",

    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative"
]
        
model = Naive_Bayes()

model.fit(X_train, y_train)
X_test = [
    "iron man is fantastic and exciting",
    "thor has a boring story",
    "captain america is an amazing hero",
    "avengers has disappointing action",
    "spider man is wonderful and entertaining",
    "doctor strange has terrible visuals",
    "black panther is inspiring",
    "guardians of the galaxy is boring"
]

predictions = model.predict(X_test)

print(predictions)