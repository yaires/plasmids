 
import numpy as np
from math import * 
import matplotlib.pyplot as plt
import pandas as pd
import random
import networkx as nx
import timeit
import pickle
import time

import csv

trim_length=800
remove_edges_pred=0.15

dset={0:'E8867',1:'E0139',2:'E1334'}
dtype={0:'connections.tab',1:'nodes_coverage.tab',2:'nodes_length.tab',3:'posterior_probabilities.tab',4:'correct_paths.csv'}

data=[[0]*len(dtype)]*len(dset)

for i in range(len(dset)):
    dt=[0]*len(dtype)
    for j in range(len(dtype)):
        dt[j]=list(csv.reader(open('C:\\Users\Gebruiker\Documents\Python Scripts\\' +dset[i]+ '\\' +dtype[j], "r")))
        print(len(dt[j]))
    data[i]=dt


graphs=[{}]*len(dset)
attributes=[np.zeros([len(dtype),len(data[i][1])-1]) for i in dset]
nodenr=[{}]*len(dset)
nodeind=[{}]*len(dset)
shft={0:1,1:1,2:1,3:0}

for i in range(len(dset)):
    nodenri={}
    nodeindi={}
    for k in range(len(data[i][1])-1):
        for j in range(3):
            attributes[i][j,k]=float(data[i][j+1][k+1][0].split()[1].split('"')[shft[j+1]])
            nodenri[k]=data[i][1][k+1][0].split()[0]
            nodeindi[data[i][1][k+1][0].split()[0]]=k
    nodenr[i]=nodenri
    nodeind[i]=nodeindi

edges_matrix=[np.zeros([len(data[i][1])-1,len(data[i][1])-1]) for i in dset]
edges=[{}]*len(dset)
for i in range(len(dset)):
    edgesi={}
    for n in range(1,len(data[i][0])):
        node1=nodeind[i][data[i][0][n][0].split()[0]]
        edgesi[node1]=[]
    for n in range(1,len(data[i][0])):
        node1=nodeind[i][data[i][0][n][0].split()[0]]
        node2=nodeind[i][data[i][0][n][0].split('"')[1]]
        edges_matrix[i][node1,node2]=1
        edgesi[node1].append(node2)
    edges[i]=edgesi

def find_paths2(graph,k,size,paths,path=[]):
    path=path+[k]
    if not (k in graph):
        paths.append(path) 
    elif len(path)>=size:
         paths.append(path)
    else:
        for node in graph[k]:
            if node in path:
                paths.append(path)
            else:
                find_paths2(graph,node,size,paths,path)
        
def find_allpaths(graph,size):
    paths=[]
    tic=timeit.default_timer()
    for node in graph:
        find_paths2(graph,node,size,paths)
    toc=timeit.default_timer()
    print(toc-tic)
    return paths

solutions_contigs=[0]*len(dset)
solutions=[0]*len(dset)
for i in range(len(dset)):
    solutions_contigs[i]=[nodeind[i][node] for loop in data[i][-1] for node in loop[1:]]
    solutions[i]=[[nodeind[i][node] for node in loop[1:]] for loop in data[i][-1]]
for i in range(len(dset)):
    for k in solutions_contigs[i]:
            attributes[i][3,k]=1

#for i in dset:
#    print([sum([attributes[i][1,node] for node in loop]) for loop in solutions[i]])


xlen=np.log(np.concatenate([attributes[i][1] for i in dset]))/np.log(10)
xcov=np.concatenate([attributes[i][0] for i in dset])
xpred=np.concatenate([attributes[i][2] for i in dset])
xsol=np.concatenate([attributes[i][3] for i in dset])


def mean_pred(i,path):
    """
    mean=1
    for node in path:
        mean=(attributes[i][2,node])*mean
        mean=mean**(1/len(path))
        #mean=1-mean
    """
    mean=0
    for node in path:
        if attributes[i][2,node]> mean:
            mean=attributes[i][2,node]
    """
    mean=0
    for node in path:
        mean=mean+attributes[i][2,node]/len(path)
    """
    
    return mean

def total_length(i,path):
    length=0
    for node in path:
        length=length+attributes[i][1,node]
    return length

def total_cov(i,path):
    cov=0
    for node in path:
        cov=cov+attributes[i][0,node]
    return cov

def trim(graph,i):
    for B in graph:
        if (attributes[i][1,B]<trim_length):
            for C in [A for A in graph if B in graph[A]]:
                graph[C].remove(B)
                for D in graph[C]+graph[B]:
                    if D not in graph[C]:
                        graph[C]=graph[C]+[D]
            graph[B]=[]
    return graph

def remove_edges(graph,i):
    graph2={}
    for node1 in graph:
        graphout=[]
        for node2 in graph[node1]:
            if mean_pred(i,[node2,node1])>remove_edges_pred:
                graphout=graphout+[node2]
        print(graphout)
        graph2[node1]=graphout
    return graph2
    

def plot_paths():
    for i in dset:
        trim(edges[i],i)
        soledges=[]
        solutiontrim=[[]]*len(solutions[i])
        for n in range(len(solutions[i])):
            solutiontrimn=[]
            for k in solutions[i][n]:
                if attributes[i][1,k]>trim_length:
                    solutiontrimn.append(k)
            solutiontrim[n]=solutiontrimn
        
        for solution in solutiontrim:
            if len(solution)>3:
                for j in range(len(solution)):
                    soledges.append([solution[j],solution[(j+1)%len(solution)],solution[(j+2)%len(solution)],solution[(j+3)%len(solution)]])
        print(solutiontrim)
        alledges=find_allpaths(edges[i],4)
        xxpred=np.array([mean_pred(i,path) for path in alledges])
        xxlen=np.log(np.array([total_length(i,path) for path in alledges]))/np.log(10)
        xxsol=np.array([(edge in soledges) for edge in alledges])*1
        xxcov=np.array([total_cov(i,path) for path in alledges])
        
        plt.scatter(xxlen,xxpred,c=xxcov,s=-6*(xxsol-1))
        
        xspred=np.array([mean_pred(i,path) for path in soledges])
        xslen=np.log(np.array([total_length(i,path) for path in soledges]))/np.log(10)
        xssol=np.array([(edge in soledges) for edge in soledges])*1
        xscov=np.array([total_cov(i,path) for path in soledges])
        plt.scatter(xslen,xspred,c=xscov,s=86*xssol)
        
'''
for i in range(len(dset)):
    gs=graphs[i]
    for n in range(1,len(data[i][0])):
        node1=data[i][0][n][0].split()[0]
        node2=data[i][0][n][0].split('"')[1]
        gs[node1].append(node2)
    graphs[i]=gs


cutoff_probability=0.8
node_length={}
node_cov={}
node_pred={}


for i in range(len(dnl)-1):
    node=dnl[i+1][0].split()[0]
    length=int(dnl[i+1][0].split()[1].split('"')[1])
    cov=float(dnc[i+1][0].split()[1].split('"')[1])
    pred=float(dp[i+1][0].split()[1])
    node_length[node]=length
    node_cov[node]=cov
    node_pred[node]=pred
    
x=np.zeros(int(len(node_length)/2))
y=np.zeros(int(len(node_length)/2))
z=np.zeros(int(len(node_length)/2))

for i in range(len(x)-1):
    z[i]=node_pred[str(i+1)+'+']
    x[i]=node_length[str(i+1)+'+']
    y[i]=node_cov[str(i+1)+'+']
 

def plot_contigs(xlen,xcov,xpred,xsol):
    plt.title("coverage spread for several contigs")
    plt.xlabel('length 10log')
    plt.ylabel('coverage')
    plt.scatter(xlen,xpred,c=xcov,s=86*xsol)
    plt.scatter(xlen,xpred,c=xcov,s=-12*(xsol-1))
    plt.show()
'''
'''
 352: [322, 361, 260, 310, 238, 288, 360, 275, 179, 233],
 353: [282, 46, 323, 334, 299, 281, 292],
 354: [25, 263, 286, 348, 69, 324, 245, 311, 339, 379, 347, 234, 272],
 355: [341],
 356: [356, 243],
 357: [357, 235],
 358: [250, 285],
 359: [284, 302],
 360: [353,
  178,
  274,
  273,
  280,
  257,
  222,
  305,
  345,
  25,
  263,
  286,
  348,
  282,
  69,
  324,
  245,
  311,
  339,
  379,
  347,
  234,
  272,
  46,
  323,
  334,
  299,
  281,
  292],
 361: [353,
  178,
  274,
  273,
  280,
  257,
  222,
  305,
  345,
  25,
  263,
  286,
  348,
  282,
  69,
  324,
  245,
  311,
  339,
  379,
  347,
  234,
  272,
  46,
  323,
  334,
  299,
  281,
  292],
 362: [],
 363: [],
'''