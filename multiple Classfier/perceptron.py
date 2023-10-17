import numpy as np

# Perceptron 객체

# 알고리즘이 iterative 하다: gradient descent와 같이 결과를 내기 위해서 여러 번의
#                           최적화 과정을 거쳐야 되는 알고리즘
# batch size: 전체 트레이닝 데이터 셋을 여러 작은 그룹으로 나누었을 때 batch size는
#            하나의 소그룹에 속하는 데이터의 수를 의미한다. 전체 트레이닝 셋을 작게 나누는 이유는 트레이닝 데이터를 통째로 신경망에 넣으면 비효율적이
#            리소스 사용으로 학습 시간이 오래 걸리기 때문입니다.
# epoch : 딥러닝에서 epoch는 전체 트레이닝 셋이 신경망을 통과한 횟수를 의미한다.


class Perceptron(object):
    '''Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    Attributes

    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    '''

    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter


# 데이터 학습시키기

    def fit(self, X, y):
        '''
        Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples
        is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        '''

        self.w_ = np.random.normal(0, 1, (3, X.shape[1]))
        self.b = np.zeros((3, 1))

        r = y.shape[0]
        y_hot = np.zeros((r, 3))
        for i in range(r):
            y_hot[i, y[i]] = 1

        for _ in range(self.n_iter):

            w_grad = np.zeros((3, X.shape[1]))

            b_grad = np.zeros((3, 1))

            for x, y in zip(X, y_hot):
                errors = 0
                z = np.dot(x, self.w_.T)+self.b.T
                z = np.clip(z, -100, None)  # Nan 값 방지

                exp_z = 1/(1+np.exp(-z))

                a = exp_z / np.sum(exp_z)  # softmax

                errors += -np.sum(y*np.log(a))

                w_grad += -np.dot(x.reshape(-1, 1), (y-a)).T

                b_grad += -(y-a).T
            self.errors_ = errors/len(y_hot)
            self.w_ -= self.eta * (w_grad/len(y_hot))
            self.b -= self.eta*(np.array(b_grad)/len(y_hot))

        return self
# 데이터 예측하기

    def predict(self, X):

        z = np.dot(X, self.w_.T)+self.b.T
        print("오차:", self.errors_)
        return np.argmax(z, axis=1)  # 가중치 1 ,2 ,3 을 적용했을때 나오는 가장 높은 확률을 선택한다.
