import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

TRAINING_SET = "spambase/spambase.data"

#Reading the Data file
def load_csv(filename):
    fread = open(filename, "r")
    data = np.loadtxt(fread, delimiter=",")
    return data


#transforming it to the TFIDF representation
def tfidf(mail):
    ndoc = mail.shape[0]
    idf = np.log10(ndoc / (mail != 0).sum(0))
    return mail / 100.0 * idf

#loading the Data files for CSV file
mail = load_csv(TRAINING_SET)
np.random.shuffle(mail)

#Taking the required classes and prediction class
Y = mail[:, 57] # classes
X = mail[:, :54] # values

X = tfidf(X)


#Classify using Linear Kernel and measuring the accuracy
#also doing the cross validation for 10 times
svm_linear = SVC(kernel="linear", C = 1.0)
acc_linear = cross_val_score(svm_linear, X, Y, cv = 10, n_jobs= 8)
print("Min Accuracy Linear Kernel: " + str(acc_linear.min()) + "\n")
print("Max Accuracy Linear Kernel: " + str(acc_linear.max()) + "\n")
print("=================================")


#Classify using Polynomial Kernel with degree 2 and measuring the accuracy
#also doing the cross validation for 10 times
svm_poly2 = SVC(kernel="poly", degree = 2, C = 1.0)
acc_poly2 = cross_val_score(svm_poly2, X, Y, cv = 10, n_jobs= 8)

print("Min Accuracy Polynomial  Kernel: " + str(acc_poly2.min()) + "\n")
print("Max Accuracy Polynomial Kernel: " + str(acc_poly2.max()) + "\n")
print("=================================")


#Classify using Gaussian Radial Basis Function Kernel and measuring the accuracy
#also doing the cross validation for 10 times
svm_rbf = SVC(kernel="rbf", C = 1.0)
acc_rbf = cross_val_score(svm_rbf, X, Y, cv = 10, n_jobs= 8)
print("Min Accuracy RBF Kernel: " + str(acc_rbf.min()) + "\n")
print("Max Accuracy RBF Kernel: " + str(acc_rbf.max()) + "\n")
print("=================================")

#Transforming above kernels to the angular kernels
norms = np.sqrt(((X+1e-128) ** 2).sum(axis=1, keepdims=True))
AX = np.where(norms > 0.0, X / norms, 0.)

#angular Linear kernel
angular_linear = SVC(kernel="linear", C = 1.0)
acc_angular_linear = cross_val_score(angular_linear, AX, Y, cv = 10, n_jobs= 8)
print("Min Accuracy Angular Linear Kernel: " + str(acc_angular_linear.min()) + "\n")
print("Max Accuracy Angular Linear Kernel: " + str(acc_angular_linear.max()) + "\n")
print("=================================")

#angular polynomial 2 kernel
angular_poly2 = SVC(kernel="poly", degree = 2, C = 1.0)
acc_angular_poly2 = cross_val_score(angular_poly2, AX, Y, cv = 10, n_jobs= 8)
print("Min Accuracy Angular Poly 2 Kernel: " + str(acc_angular_poly2.min()) + "\n")
print("Max Accuracy Angular Poly 2 Kernel: " + str(acc_angular_poly2.max()) + "\n")
print("=================================")

#angular gaussian radial basis function kernel
angular_rbf = SVC(kernel="rbf", C = 1.0)
acc_angular_rbf = cross_val_score(angular_rbf, AX, Y, cv = 10, n_jobs= 8)
print("Min Accuracy Angular RBF Kernel: " + str(acc_angular_rbf.min()) + "\n")
print("Max Accuracy Angular RBF Kernel: " + str(acc_angular_rbf.max()) + "\n")
print("=================================")


#Learning the models
from sklearn.model_selection import train_test_split

#Setting the criteria for testing data and training data
#models for simple kernels
#keeping the test size 30% and for traing 70%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#models for Angular Kenels
ax_train, ax_test, ay_train, ay_test = train_test_split(AX, Y, test_size=0.3)
n = AX.shape[0]
n_train = ax_train.shape[0]
print(n_train)

#print the models
def print_model(clf, xx_train, yy_train):
    clf_fit = clf.fit(xx_train, yy_train)
    print("----------------")
    print(str(clf_fit.score(xx_train, yy_train)))
    print("----------------")
    print(str(clf_lna.n_support_)) # number of support vectors for each class
    print("----------------")
    print(str(clf_lna.support_vectors_)) # printing support_vectors
    print("----------------")

angular_linear = SVC(kernel="linear", C=1)
clf_lna = angular_linear.fit(ax_train, ay_train)
print(str(angular_linear.score(ax_train, ay_train)))

print("\n")
print("Printing--------Linear Kernel------")
print("\n")

#printing the linear kernel
clf_ln = svm_linear.fit(x_train, y_train)
print(str(svm_linear.score(x_train, y_train)))
print(str(clf_ln.support_vectors_.shape))
print(str(clf_ln.n_support_))
print(str(clf_lna.support_vectors_[0]))

print("\n")
print("Printing--------Angular Linear Kernel------")
print("\n")
#printing the Angular linear Kernel

print(str(clf_lna.support_vectors_.shape))
print(str(clf_lna.n_support_))
print(clf_lna.support_vectors_.shape[0]/n_train)
print(str(clf_lna.support_vectors_[0]))



print("\n")
print("Printing--------Polynomial Kernel------")
print("\n")

# Polynomial kernel parameters
print_model(svm_poly2, ax_train, ay_train)

print("\n")
print("Printing--------Angular Polynomial Kernel / Polynomial Kernel------")
print("\n")
# Polynomial angular kernel parameters
print_model(angular_poly2, ax_train, ay_train)
# Note that for standard polynomial and angular polynomial we obtain the same parameters and results

print("\n")
print("Printing--------RBF Kernel------")
print("\n")
# RBF kernel parameters
print_model(svm_rbf, ax_train, ay_train)

print("\n")
print("Printing--------RBF Angular Kernel------")
print("\n")
# Angular RBF kernel parameters
print_model(angular_rbf, ax_train, ay_train)
