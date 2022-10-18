import random

Dt = 0
I = 1000

Date = []
Dp = []
Droi = []

# get random values from 1%-3% 
def get_random_roi():
    return random.uniform(0.01, 0.03)

for d in range(1, 365):
    Dt = d
    ROI = get_random_roi()
    I = I + (I * ROI)
    Droi.append(ROI)
    Date.append(Dt)
    Dp.append(I)

import seaborn as sns
import matplotlib.pyplot as plt

# plot Date,Dp and Droi sub graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(Date, Dp, 'b')
ax1.set_title('Profit per day')
ax1.set_xlabel('Day')
ax1.set_ylabel('Profit in (R)')
ax1.ticklabel_format(useOffset=False, style='plain')
#add legend
ax1.legend(['Dp'])
ax2.plot(Date, Droi, 'r')
ax2.legend(['Droi'])
ax2.set_title('ROI per day')
ax2.set_xlabel('Day')
ax2.set_ylabel('ROI in %')
ax2.ticklabel_format(useOffset=False, style='plain')
#save plot
fig.savefig('images/plot.png')



# curve fit Dp and Droi
import numpy as np
from scipy.optimize import curve_fit

def func(Dp, T):
    return 1000*np.exp(T*Dp)

xdata = np.array(Date)
ydata = np.array(Dp)

popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)

#print equation
print('Dt = 1000 * exp(T * Dp)')
# print Dt at 1
equation = 'Dt = 1000 * exp('+str(round(popt[0],2))+'t)'

# plot curve fit Dp and Droi sub graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(xdata, ydata, 'b', label='data')
ax1.plot(xdata, func(xdata, *popt), 'r-', label='fit')
ax1.set_title('Profit per day')
ax1.set_xlabel('Day')
ax1.set_ylabel('Profit in (R)')
ax1.ticklabel_format(useOffset=False, style='plain')
#add legend
ax1.legend(['Actual Profit at time t', equation])
ax2.plot(Date, Droi, 'r')
ax2.legend(['Droi'])
ax2.set_title('ROI per day')
ax2.set_xlabel('Day')
ax2.set_ylabel('ROI in %')
ax2.ticklabel_format(useOffset=False, style='plain')
plt.show()

#save plot
fig.savefig('images/fit.png')









