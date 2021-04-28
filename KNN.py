import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

TRAINING_SET = "spambase/spambase.data"



#Reading the Data file
def load_csv(filename):
    fread = open(filename, "r")
    data = np.loadtxt(fread, delimiter=",")
    return data

#loading the Data files for CSV file
mail = load_csv(TRAINING_SET)
np.random.shuffle(mail)

#Taking the required classes and prediction class
Y = mail[:, 57] # classes
X = mail[:, :54] # values

#Learning by the model
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print("KNN with K=3")
print("\n")
#Classifying using KNN with K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_KNN = cross_val_score(knn, X, Y, cv = 10, n_jobs= 8)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Min Accuracy 3NN: " + str(acc_KNN.min()) + "\n")
print("Max Accuracy 3NN: " + str(acc_KNN.max()) + "\n")
print("Mean Accuracy 3NN: " + str(acc_KNN.mean()) + "\n")
print("Variance Accuracy 3NN: " + str(acc_KNN.var()) + "\n")
print("=================================")

print("KNN with K=5")
print("\n")
#Classifying using KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_KNN = cross_val_score(knn, X, Y, cv = 10, n_jobs= 8)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Min Accuracy 5NN: " + str(acc_KNN.min()) + "\n")
print("Max Accuracy 5NN: " + str(acc_KNN.max()) + "\n")
print("Mean Accuracy 5NN: " + str(acc_KNN.mean()) + "\n")
print("Variance Accuracy 5NN: " + str(acc_KNN.var()) + "\n")
print("=================================")

print("KNN with K=7")
print("\n")
#Classifying using KNN with K=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_KNN = cross_val_score(knn, X, Y, cv = 10, n_jobs= 8)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Min Accuracy 7NN: " + str(acc_KNN.min()) + "\n")
print("Max Accuracy 7NN: " + str(acc_KNN.max()) + "\n")
print("Mean Accuracy 7NN: " + str(acc_KNN.mean()) + "\n")
print("Variance Accuracy 7NN: " + str(acc_KNN.var()) + "\n")
print("=================================")


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train,y_train)
print(pca.components_)

print(pca.explained_variance_)

pca = PCA(n_components=1)
pca.fit(X_train,y_train)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)



X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');