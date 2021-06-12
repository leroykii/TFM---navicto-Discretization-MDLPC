from fxpmath import Fxp
import matplotlib.pyplot as plt
import numpy as np

x1 = Fxp(0.5, signed=True, n_word=3, n_frac=1)
x2 = Fxp(1.5, signed=True, n_word=3, n_frac=1)

# numpy.single

array = np.single(20.9) # Or np.float32(20.9)

print(array)

y = Fxp(None, signed=True, n_word=4, n_frac=2)
f = 5.0         # signal frequency
fs = 400.0      # sampling frequency
N = 1000        # number of samples

n = Fxp( list(range(N)) )                       # sample indices
y( 0.5 * np.sin(2 * np.pi * f * n() / fs) )  

y.info(verbose=3)

x = np.linspace(0, 1, N)
plt.plot(x, y, label='linear')

# Add a legend
plt.legend()

# Show the plot
# plt.show()

#########################

np1 = np.single(20.9) # 
fxp1 = Fxp(1.5, signed=True, n_word=3, n_frac=1)

res = Fxp(None, signed=True, n_word=12, n_frac=2)

res.equal(np1+fxp1)
print(res)

res.info(verbose=3)