from fxpmath import Fxp
import matplotlib.pyplot as plt
import numpy as np

result = []

for i in range(0, 31): 
    TEMPLATE = Fxp(None, True, 32, i)    
    pi_fxp = Fxp(np.pi).like(TEMPLATE)
    print(pi_fxp)
    result.append(pi_fxp)

expected = np.ones(31)*np.pi

precision_error = np.single(result) - expected
print(precision_error)

print(result)


########### Plot result

plt.subplot(2, 1, 1)
plt.plot(result)
plt.plot(expected)

# Add a legend
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(precision_error)

# Show the plot
plt.show()
