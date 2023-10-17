import matplotlib.pyplot as plt
import cp_utils as cpu
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from perceptron import Perceptron

LF = '\n'

iris = datasets.load_iris()

# check the target features of the dataset
# print(iris['target_names'])
# print(iris['target'])

# see the descriptive features of the sample
# print(iris['data'])
# print(iris['data'].shape)

X = iris.data[:, (0, 2)]
y = iris.target
#print(X, '\n', y)

# ----------------------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)  # 랜덤 데이터 추출 옵션

# ----------------------------------------------

# 데이터 전처리
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#print(X_train_std, LF, X_train)

# ----------------------------------------------


#ppn = Perceptron(max_iter=10, eta0=0.1, random_state=0)
ppn = Perceptron(0.1, 100)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)


print('Misclassified samples: %d' % (y_test != y_pred).sum())

# ----------------------------------------------

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

cpu.plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

print('End of Program')
