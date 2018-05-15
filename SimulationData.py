import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
np.random.seed(0)

class SimulationData(object):
    """
        Datasets: leftdown, center, double_center, half_moon, two_spirals
        Method: plot
    """
    np.random.seed(0)
    def __init__(self, total_num, minor_num, rs=8):
        self.total_num = total_num
        self.minor_num = minor_num
        self.rs = rs

    def plot(self, data):
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                              int(max(data[1]) + 1))))
        plt.scatter(data[0][:, 0], data[0][:, 1], s=10, color=colors[1-data[1]])
        plt.show()

    @property
    def leftdown(self):
        blobs = np.random.uniform([0,0],[2,2],
                                  size=[self.minor_num,2]), np.repeat(0,self.minor_num)
        no_structure = np.random.rand(self.total_num, 2)*12, None
        data1 = [[i,j] for i,j in zip(no_structure[0][:,0],no_structure[0][:,1]) if not ((0<=i<=2) and (0<=j<=2))]
        data1 = np.array(data1)
        data1 = data1,np.repeat(1,data1.shape[0])
        data = np.concatenate((data1[0],blobs[0])),np.concatenate((data1[1],blobs[1]))
        return data

    @property
    def center(self):
        blobs = np.random.uniform([5,5],[7,7],
                                  size=[self.minor_num,2]), np.repeat(0,self.minor_num)
        no_structure = np.random.rand(self.total_num, 2)*12, None
        data1 = [[i,j] for i,j in zip(no_structure[0][:,0],no_structure[0][:,1]) if not ((5<=i<=7) and (5<=j<=7))]
        data1 = np.array(data1)
        data1 = data1,np.repeat(1,data1.shape[0])
        data = np.concatenate((data1[0],blobs[0])),np.concatenate((data1[1],blobs[1]))
        return data

    @property
    def double_center(self):
        blobs = datasets.make_blobs(n_samples=self.minor_num, random_state=self.rs, centers= [[10,10], [7,7]],cluster_std=0.4)
        no_structure = np.random.rand(self.total_num, 2)*9+3, None
        data1 = [[i,j] for i,j in zip(no_structure[0][:,0],no_structure[0][:,1]) if not ((5.7<=i<=8.3) and (5.7<=j<=8.3))]
        data1 = np.array(data1)
        data1 = [[i,j] for i,j in zip(data1[:,0],data1[:,1]) if not ((9<=i<=11) and (9<=j<=11))]
        data1 = np.array(data1)
        data1 = data1,np.repeat(1,data1.shape[0])
        labels = np.repeat(0,len(blobs[1]))
        data = np.concatenate((data1[0],blobs[0])),np.concatenate((data1[1],labels))
        return data

    @property
    def half_moon(self):
        noisy_moons = datasets.make_moons(n_samples=self.total_num, noise=.02,random_state=self.rs)
        noisy_moons1 = datasets.make_moons(n_samples=self.minor_num, noise=.03, random_state=8)
        data_moon = noisy_moons[0][noisy_moons[1]==1,:],np.repeat(1,int(self.total_num/2))
        data_moon1 = 0.8-noisy_moons1[0][noisy_moons1[1]==0,:]+[0.23,0],np.repeat(0,int(self.minor_num/2))
        data = np.concatenate((data_moon[0],data_moon1[0])),np.concatenate((data_moon[1],data_moon1[1]))
        return data

    @property
    def two_spirals(self):
        n = np.sqrt(np.random.rand(self.total_num,1)) * 780 * (2*np.pi)/360
        d1x = -np.cos(n)*n + np.random.rand(self.total_num,1) * 0.5
        d1y = np.sin(n)*n + np.random.rand(int(self.total_num),1) * 0.5
        X,y = (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
                np.hstack((np.zeros(self.total_num),np.ones(self.total_num))))
        X1 = np.array((X[y==0,0][:self.minor_num], X[y==0,1][:self.minor_num])).transpose()  #######use Transpose instead of reshape!!!!!
        X2 = np.array((X[y==1,0], X[y==1,1])).transpose()
        X_12 = np.concatenate((X1,X2))
        y_12 = np.concatenate((np.repeat(0,self.minor_num),np.repeat(1,1000)))
        data = X_12,y_12
        return data
