#Importing required modules
 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Load Data
def Clustering(Genre, metadata): # metadata has 9 features (dimensions) Loudness, acousticness, danceablitiy, energy, instrumentalness, liveness, speechness, tempo, valence
    
    
    def ClusterIndices(clustNum, labels_array): # get index from all samples in one cluster
        return np.where(labels_array == clustNum)[0]

    def Plot():  
        #plotting the results:
        ax = plt.axes(projection ='3d')
        for i in u_labels:
            ax.scatter(df[label == i , 0] , df[label == i , 1] , df[label == i , 2], label = i)
       
        plt.legend()
        plt.show()
        return None

# # assigning Genres to Clust numbers
#     if Genre == "Metal":    # Check Number of Clust Num Or change Clust Num to String
#         ClustNum = 1
#     elif Genre == "k-pop":
#         ClustNum = 2    
        
    NUM_GENRES # number of Genres in dataset
    
    data = MinMaxScaler().transform(metadata)

    pca = PCA(3)
    
    #Transform the data
    df = pca.fit_transform(data)

    
    #Initialize the class object
    kmeans = KMeans(n_clusters= NUM_GENRES)
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    #Getting unique labels
    u_labels = np.unique(label)
    
    
    Plot()

    Indices = ClusterIndices(ClustNum, kmeans.labels_)  # Need to assign genres to Clust numbers, Clust Num = "Genre", create tabelle für ClustNum and Genre
    return Indices

    


