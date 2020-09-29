# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:13:04 2020

@author: Pedro
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import sys
import cv2 as cv
from skimage import filters
import time 
from PIL import Image
from scipy import ndimage


x=[10.0,8.0,13.0,9.0,11.0,14.0,6.0,4.0,12.0,7.0,5.0]
x4=[8.0,8.0,8.0,8.0,8.0,8.0,8.0,19.0,8.0,8.0,8.0]

y1=[8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]
y2=[9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74]
y3=[7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73]
y4=[6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]

x_points=[16,8,16,16]
y_points=[8,6,10,6]


plt.plot(x,y1 ,'ro',label='Table 1')
plt.plot(x,y2 ,'bo',label='Table 2')
plt.plot(x,y3 ,'go',label='Table 3')
plt.plot(x4,y4 ,'mo',label='Table 4')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.savefig("Problema teoría 1.png")
plt.show()

plt.plot(x,y1 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.show()
plt.plot(x,y2 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.show()
plt.plot(x,y3 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.show()
plt.plot(x4,y4 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.show()

"""
print(np.mean(x))
p=[]
for i in np.arange(0,4):
    p=np.append(x,x_points[i])
    print(p)
    print(np.mean(p))

print('-------')

for i in np.arange(0,4):
    p=np.append(x4,x_points[i])
    print(p)
    print(np.mean(p))
    
print('-------')   
for i in np.arange(0,4):
    p=np.append(y3,y_points[i])
    print(p)
    print(np.mean(p))
"""

def func2(x,a,b,c):
    return a*x**2+b*x+c

popt, pcov = scipy.optimize.curve_fit(func2, x, y2)


print('Valores ajuste (ax^2+bx+c).   a=',popt[0],'   b=',popt[1], 'c=',popt[2])

x2_ajuste = np.linspace(3,17)
y2_ajuste = popt[2]+popt[1]*(x2_ajuste)+popt[0]*(x2_ajuste)**2

plt.plot(x2_ajuste,y2_ajuste ,'red',label='Ajuste (ax^2+bx+c)')
plt.plot(x,y2 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()

#perr = np.sqrt(np.diag(pcov))
#print(perr)


def func4(x,a,b,c,d):
    return a*np.exp(b*x+c)+d

popt, pcov = scipy.optimize.curve_fit(func4, x4, y4)


print('Valores ajuste (a*ln(b*x+c).   a=',popt[0],'   b=',popt[1], 'c=',popt[2],'d=',popt[3])

x4_ajuste = np.linspace(0,20)
y4_ajuste = popt[0]*np.exp(popt[1]*x4_ajuste+popt[2])+popt[3]

plt.plot(x4_ajuste,y4_ajuste ,'red',label='Ajuste (a*ln(b*x+c)')
plt.plot(x4,y4 ,'ro',label='Table 1')
plt.plot(x_points,y_points ,'ko',label='Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()

#perr = np.sqrt(np.diag(pcov))
#print(perr)
