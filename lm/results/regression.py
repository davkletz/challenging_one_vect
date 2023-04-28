from joblib import load as ld
from sklearn.linear_model import LinearRegression
import numpy as np

x = ld("freqs.joblib")
y = ld("norms.joblib")


clf = LinearRegression()

clf.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1), sample_weight=x)


print(clf.coef_)
print(clf.intercept_)

