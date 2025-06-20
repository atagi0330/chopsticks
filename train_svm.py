import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import joblib
from sklearn import svm

with open("X.txt", mode="r") as fx:
    X1 = [eval(line.strip()) for line in fx]

with open("Y.txt", mode="r") as fy:
    y = [int(line.strip()) for line in fy]

X1 = np.array(X1)
y = np.array(y)

pre_X = np.zeros((len(X1), 21, 3))
X = np.zeros((len(X1), 18))

for i in range(len(X1)):
    pre_X[i, :, :] = X1[i].reshape(21, 3)


# 0(手首)の座標で正規化
for i in range(len(X1)):
    pre_X[i, :, :] = pre_X[i, :, :] - pre_X[i, 0, :]


# 指の角度を算出
for i in range(len(X1)):
    x = []
    for j in range(5):
        x0 = pre_X[i, 0, :]
        x1 = pre_X[i, 4 * j + 1, :]
        x2 = pre_X[i, 4 * j + 2, :]
        x3 = pre_X[i, 4 * j + 3, :]
        x4 = pre_X[i, 4 * j + 4, :]
        if j == 0:
            x.append(np.dot(x0 - x2, x3 - x2) / np.linalg.norm(x0 - x2) / np.linalg.norm(x3 - x2))
        else:
            x.append(np.dot(x0 - x1, x2 - x1) / np.linalg.norm(x0 - x1) / np.linalg.norm(x2 - x1))
            x.append(np.dot(x1 - x2, x3 - x2) / np.linalg.norm(x1 - x2) / np.linalg.norm(x3 - x2))
        x.append(np.dot(x2 - x3, x4 - x3) / np.linalg.norm(x4 - x3) / np.linalg.norm(x2 - x3))

    for j in range(len(x)):
        X[i, j] = x[j]


# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# svmモデルの構築
model = SVC(kernel="linear", C=1.0, random_state=42)
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'svm.joblib')
joblib.dump(scaler, 'svm_scaler.joblib')
