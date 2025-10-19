from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump

#Coleta de Amostras
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = SVC(kernel='rbf', probability=False)
clf.fit(X_train, y_train)
print("Val acc:", clf.score(X_val, y_val))
dump(clf, "Amostras.joblib")