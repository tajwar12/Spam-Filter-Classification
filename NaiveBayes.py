import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

TRAINING_SET = "spambase/spambase.data"

def load_csv(filename):
    fread = open(filename, "r")
    data = np.loadtxt(fread, delimiter=",")
    return data

mail = load_csv(TRAINING_SET)


class NaiveBayesClassifier(BaseEstimator):

    def score(self, X, Y):
        p_x_spam_i = (2 * np.pi * self.var_spam) ** (-1. / 2) * np.exp(
            -1. / (2 * self.var_spam) * np.power(X - self.mu_spam, 2))
        p_x_ham_i = (2 * np.pi * self.var_ham) ** (-1. / 2) * np.exp(
            -1. / (2 * self.var_ham) * np.power(X - self.mu_ham, 2))

        p_x_spam = np.prod(p_x_spam_i, axis=1)
        p_x_ham = np.prod(p_x_ham_i, axis=1)

        p_spam_x = p_x_spam * self.p_spam
        p_ham_x = p_x_ham * self.p_ham

        predicted_labels = np.argmax([p_ham_x, p_spam_x], axis=0)
        return np.mean(predicted_labels == Y)

    def fit(self, X, Y, **kwargs):
        self.spam = X[Y == 1, :54]
        self.ham = X[Y == 0, :54]

        self.N = float(self.spam.shape[0] + self.ham.shape[0])
        self.k_spam = self.spam.shape[0]  # frequency of spam
        self.k_ham = self.ham.shape[0]  # frequency of ham

        self.p_spam = self.k_spam / self.N
        self.p_ham = self.k_ham / self.N

        self.mu_spam = np.mean(self.spam, axis=0)
        self.mu_ham = np.mean(self.ham, axis=0)

        # Avoid division by zero adding a small costant
        self.var_spam = np.var(self.spam, axis=0) + 1e-128
        self.var_ham = np.var(self.ham, axis=0) + 1e-128

# shuffling the dataset
np.random.shuffle(mail)
#assigning labels
Y = mail[:, 57]
X = mail[:, :54]

#10 way cross validation
scores = cross_val_score(NaiveBayesClassifier(), X, Y, cv = 10)

print("Min Accuracy: " + str(scores.min())+"\n")
print("Mean Accuracy: " + str(scores.mean())+"\n")
print("Max Accuracy: " + str(scores.max())+"\n")
print("Variance/Std Accuracy: " + str(scores.var()) +" / " +str(scores.std())+"\n")

print("=================================")


# Apply 10-Way Cross validation 'run' times and get all the scores
def eval_model(data, classifier, run = 10):
    scores = np.array([])
    for i in range(run):
        np.random.shuffle(data)
        Y = mail[:, 57]
        X = mail[:, :54]
        scores = np.append(scores,cross_val_score(classifier, X, Y, cv = 10))
    return scores

scores_run = eval_model(mail, NaiveBayesClassifier(), run = 20)
print("Min Accuracy: " + str(scores_run.min())+"\n")
print("Mean Accuracy: " + str(scores_run.mean())+"\n")
print("Max Accuracy: " + str(scores_run.max())+"\n")
print("Variance/Std Accuracy: " + str(scores_run.var()) +" / " +str(scores_run.std())+"\n")
print("=================================")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


from sklearn.naive_bayes import GaussianNB
clf = NaiveBayesClassifier()
#clf = GaussianNB()
md = clf.fit(x_train,y_train)

#Accurace of the Model
print("Accuracy: "+str(clf.score(x_test, y_test)))


print("p(spam): "+str(clf.p_spam))
print("p(ham): "+str(clf.p_ham))

#Modeling Parameters
#Mean and Variance of Spam Class
print("mu spam: "+str(np.round(clf.mu_spam,3)))
print("var spam: "+str(np.round(clf.var_spam,3)))

#Mean and Variance of Ham Class
print("mu ham: "+str(np.round(clf.mu_ham,3)))
print("var ham: "+str(np.round(clf.var_ham,3)))