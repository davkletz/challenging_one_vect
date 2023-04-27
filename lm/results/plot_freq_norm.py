import matplotlib.pyplot as plt
from joblib import load
import numpy as np

x = load("freqs.joblib")

y = load("norms.joblib")




print(len(x))

#print(x)
plt.semilogx(x, y, 'ro')
#plt.plot(x, y, 'ro')

plt.xlabel('Nb apparitions')
plt.ylabel('Norm')

plt.show()