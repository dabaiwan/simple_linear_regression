# 简单线性回归   simple linear regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 数据  data
rng = np.random.RandomState(42)
x = -10 * rng.rand(100)
y = -3 * x - 10 + rng.randn(100)
plt.scatter(x, y)


# 选择简单线性回归模型  Choose LinearRegression model
model = LinearRegression(fit_intercept=True)

X = x[:, np.newaxis]  # X.shape is (50,1)

model.fit(X, y)

print("斜率是", model.coef_, "，截距是", model.intercept_)

# 预测未知数据   Predict labels for unknown data
xfit = np.linspace(-11, -1)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.plot(xfit, yfit)
plt.show()

