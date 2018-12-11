import numpy as np
import os
from numpy import linalg as LA
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm as cm
from scipy import ndimage
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from matplotlib import offsetbox
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import argparse
from urllib.request import Request, urlopen
import urllib
import urllib3
import json
from pandas.io.json import json_normalize
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import scipy

def cos_metric(a, b):  # scipy.spatial.distance.cosine
    return scipy.spatial.distance.cosine(a, b)
    #return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def plot_2D_embeddings(embeddings, names, annotate=False):  # plot_2D_embeddings(embeddings, names, True)
    model = TSNE(n_components=2, perplexity=20, metric=cos_metric, init='pca', random_state=0)
    vectors = model.fit_transform(embeddings)
    x, y = vectors[:, 0], vectors[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker='.', s=2)
    i = 0
    if (annotate):
        for i, tname in enumerate(names): # .decode('unicode-escape')
                if (tname=='daxy') or (tname=='good') or  (tname=='i') or (tname=='am') or (tname=='are') or (tname=='you'):
                    ax.annotate(tname, (x[i], y[i]), color='red', arrowprops=dict(facecolor='black', shrink=0.01)) 
    else:
        pass
    plt.savefig('f2D_visualization.png')
    plt.show()

def visualize3DData (embeddings, names, n=3):  # visualize3DData(embeddings, names)
    model = TSNE(n_components=n, random_state=0)
    X =  model.fit_transform(embeddings)
    
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], depthshade = False, picker = True)


    def distance(point, event):
        assert point.shape == (3,), "Distance: point.shape is wrong: %s, must be (3,)" % point.shape
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        x3, y3 = ax.transData.transform((x2, y2))
        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)

    def calcClosestDatapoint(X, event):
        distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotatePlot(X, index):
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate( "%s" % names[index],
            xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()

    def onMouseMotion(event):
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot (X, closestIndex)

    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.savefig('f3D_Cluster.png')
    plt.show()
    
if __name__=='__main__':
    word_dictionary = pickle.load(open('daxy.orig', 'rb'))

    vocab = []
    #f = open('simwords.txt')
    #for line in f:
    #    vocab.append(line.strip())
    #print(vocab)

    X_train = []
    X_train_names = []
    for word in word_dictionary:
            X_train_names.append(word)
            X_train.append(np.asarray(word_dictionary[word].detach().numpy()))  #

    X_train = np.asarray(X_train)

    # PCA with 50 dimensions.
    pca =  PCA(n_components = 50)
    X_train = X_train - np.mean(X_train)
    X_fit = pca.fit_transform(X_train)

    X_train_plot = []
    X_train_plot_names = []
    
    for i, word in enumerate(X_train_names):
        #if word in vocab or word=='daxy':
        X_train_plot.append(X_fit[i])
        X_train_plot_names.append(word)

    plot_2D_embeddings(X_train_plot, X_train_plot_names, True)
    #visualize3DData(X_fit, X_train_names)

    
    
