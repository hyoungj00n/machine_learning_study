from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 그래프 시각화
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

raw_df = load_breast_cancer()
features = raw_df.data
labels = raw_df.target

cancer_df = pd.DataFrame(data = features, columns = raw_df.feature_names)
cancer_df['target'] = labels

cancer_df.describe()

corr = cancer_df.corr()
plt.figure(figsize=(20, 20));
sns.heatmap(corr,
            vmax=0.8,
            linewidths=0.01,
            square=True,
            annot=True,
            cmap='YlGnBu');
plt.title('Feature Correlation');

y = cancer_df['target']
X = cancer_df.drop(columns='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 156)

print('학습용 데이터셋(train data) 행 개수: {}'.format(X_train.shape[0]))
print('테스트용 데이터셋(test data) 행 개수: {}'.format(X_test.shape[0]))

# GBM 수행 시간 측정
start_time = time.time()

# 모델 설정
gb_clf = GradientBoostingClassifier(random_state = 0)
gb_clf.fit(X_train, y_train)

# GBM 학습 및 예측 성능 평가
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
print('GBM 수행 시간: {0:.1f} 초'.format(time.time() - start_time))
