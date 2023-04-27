from joblib import load as ld
from sklearn.linear_model import LinearRegression

x = ld("freqs.joblib")
y = ld("norms.joblib")


clf = LinearRegression()

clf.fit(x,y)

print(clf.coef_)