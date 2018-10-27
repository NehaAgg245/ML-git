import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0,8.0)

#Calculate h = p0+p1x
#where p0 = theta zero
#and p1 = theta one  
def h(p0, p1, xi) :
	return (p0+p1*xi);

#Calculate J error 
def JError(m,p0,p1,x,y):
	sum = 0
	for i in range(m):
		hyp = h(p0,p1,x[i])
		sum = sum + math.pow((hyp-y[i]),2)
	j = sum /(2*m)
	return j

#Calculating derivative wrt theta zero
def derZero(m,p0,p1,x,y):
	sum = 0
	for i in range(m):
		hyp = h(p0,p1,x[i])
		sum = sum + (hyp - y[i])
	j = sum / m
	return j

#Calculating derivative wrt theta one 
def derOne(m,p0,p1,x,y):
	sum = 0
	for i in range(m):
		hyp = h(p0,p1,x[i])
		sum = sum + (hyp - y[i])*x[i]
	j = sum /m
	return j 

#Calculating new params, a = step size
def pNew(pOld, a, der):
	p = pOld - a * der
	return p

x = [3,1,0,4]
y = [2,2,1,3]
a = 0.01
m = 4
p0 = 0
p1 = 1
j = []

for i in range(7):
	j.append(JError(m,p0,p1,x,y))
	dZero = derZero(m,p0,p1,x,y)
	dOne = derOne(m,p0,p1,x,y)
	p0New = pNew(p0,a,dZero)
	p1New = pNew(p1,a,dOne)
	p0 = p0New
	print ('new p0',p0)
	p1 = p1New
	print ('new p1', p1)

print (j)

