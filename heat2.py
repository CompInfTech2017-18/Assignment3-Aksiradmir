# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.constants import pi
from mpl_toolkits.mplot3d import axes3d
from scipy import linalg


a=100



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
    def __init__(self,M,N,Eta,LeftBorder,RightBorder,InitialConditions,Newton,ExternalTemperature,plotter):
        self.n=N
        self.m=M
        self.eta=Eta
        self.lb=LeftBorder
        self.rb=RightBorder
        self.tb=InitialConditions
        self.plotter=plotter
        self.h=Newton
        self.Te=ExternalTemperature
        

        
    def UnClear(self):
        A=np.zeros((self.m, self.n))
        A[:,0]=self.tb
        A[0,:]=self.rb
        A[-1,:]=self.lb
        o=np.zeros(self.m-3)
        o[:]=-1
        d=np.zeros(self.m-2)
        d[:]=2./self.eta+2.
        C=np.diag(o, k=-1)+np.diag(o, k=1)+np.diag(d,k=0)
        for j in range(self.m-1):
            B=A[:-2,j]+(2./self.eta*(1.-self.h)-2.)*A[1:-1,j]+A[2:,j]+2./self.eta*self.h*self.Te
            B[0]=B[0]+A[0,j+1]
            B[-1]=B[-1]+A[-1,j+1]
            X=np.linalg.solve(C,B)
            A[1:-1,j+1]=X
        
        x=np.linspace(0.,self.n,self.n)
        y=np.linspace(0.,self.m,self.m)
        plotter.plot(x,y,A)
        return (A)
    



IC=np.zeros(a)
IC[:]=100.
LB=np.zeros(a)
RB=np.zeros(a)

            


plotter = AdvancedPlotter('UnClear', 'x', 'y')
solveL=SolveDiff(a,a,5.,LB,RB,IC,0.,0.,plotter)
Z2=solveL.UnClear()
plotter.show()


b=a/2
IC=np.zeros(a)
IC[:50]=50.
IC[50:]=100.
LB=np.zeros(a)
RB=np.zeros(a)

plotter = AdvancedPlotter('Rod', 'x', 'y')
solveL=SolveDiff(a,a,5.,LB,RB,IC,0.,0.,plotter)
Z2=solveL.UnClear()
plotter.show()



IC=np.zeros(a)
IC[:]=100.
LB=np.zeros(a)
RB=np.zeros(a)


plotter = AdvancedPlotter(r'$T_e$=20', 'x', 'y')
solveL=SolveDiff(a,a,5.,LB,RB,IC,0.05,20.,plotter)
Z4=solveL.UnClear()
plotter.show()