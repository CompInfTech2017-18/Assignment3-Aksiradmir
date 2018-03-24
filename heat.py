# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.constants import pi
from mpl_toolkits.mplot3d import axes3d


a=100
b=500


class Plotter: 
    def __init__(self, title, xlabel, ylabel): 
        self.fig = plt.figure(facecolor='w')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title) 
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def plot(self, x, y, Mat): 
        return self.ax.pcolormesh(x, y, Mat)

    def show(self):
        plt.show()
        
class AdvancedPlotter:
    def __init__(self, title, xlabel, ylabel):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title) 
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
    def plot(self, xx, yy, Mat):
        x,y=np.meshgrid(xx,yy)
        CS=plt.contour(x,y,Mat,colors=('indigo','purple','b','m','violet','aqua'), linewidths=0.8)
        return self.ax.plot_wireframe(x, y, Mat, color='k', linewidth=0.3)

    def show(self):
        pylab.show()
        
class SolveDiff:
    def __init__(self,M,N,Eta,NumberOfIterrations,LeftBorder,RightBorder,InitialConditions,plotter):
        self.n=N
        self.m=M
        self.eta=Eta
        self.ni=NumberOfIterrations
        self.lb=LeftBorder
        self.rb=RightBorder
        self.tb=InitialConditions
        self.plotter=plotter
        

        
    def Clear(self):
        A=np.zeros((self.m, self.n))
        A[:,0]=self.tb
        A[0,:]=self.rb
        A[-1,:]=self.lb
        for i in range(self.ni):
            B=A[1:-1,:-1]+self.eta*(A[:-2,:-1]+A[2:,:-1]-2*A[1:-1,:-1])
            A[1:-1,1:]=B

        
        x=np.linspace(0.,self.n,self.n)
        y=np.linspace(0.,self.m,self.m)
        plotter.plot(x,y,A)
        return (A)
    



IC=np.zeros(a)
IC[:]=100.
LB=np.zeros(b)
RB=np.zeros(b)

            


            
plotter = AdvancedPlotter('Clear', 't', 'x')
solveL=SolveDiff(a,b,0.5,10000,LB,RB,IC,plotter)
Z1=solveL.Clear()
plotter.show()

