import numpy as np
import math as mt 
def generator(n):
    X=[]
    Y=[]
    for _ in range(n):
        u=np.random.uniform(20.0,50.0)
        angle=np.random.uniform(0.262,1.31)
        t_f=(2*u*mt.sin(angle))/9.8
        t=np.random.uniform(0,t_f)
        y=u*t*mt.sin(angle)-(0.5*9.8*t^2)
        j=[u,angle,t]
        Y.append(y)
        X.append(j)
    ar=np.array(X)
    return (ar,Y)