# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:07:52 2017

@author: zhuya
"""
import matplotlib.pyplot as plt
a = [2.121,1.082,0.6355, 0.3962, 0.266812,0.1907, 0.1355, 0.0994, 0.07539,0.05786,0.04554]
v = [2.1398,1.4143,1.1421,1.0226, 0.9639,0.934,0.9233,0.9230,0.9259,0.930,0.928]
t = [0,25,50,75,100,125,150,175,200,225,250]
plt.title('model loss xgboost')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(t,a,'b')
plt.plot(t,v,'y')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
