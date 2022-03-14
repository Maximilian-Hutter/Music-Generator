# First clustering with full data; can be easily updated with new data
 
from email.mime import audio
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from genres import genrelist
import pickle
import csv

audio_features = []

#Load Data
def clustering(audio_features): # audio_features has 9 features (dimensions) Loudness, acousticness, danceablitiy, energy, instrumentalness, liveness, speechness, tempo, valence
    
    def plot():  
        #plotting the results:
        ax = plt.axes(projection ='3d')
        for i in u_labels:
            ax.scatter(df[label == i , 0] , df[label == i , 1] , df[label == i , 2], label = i)
        
        plt.legend()
        plt.show()
        return None
       
    scaler = MinMaxScaler()
    scaler.fit(audio_features)
    data = scaler.transform(audio_features)

    pca = PCA(3) # transform data into 3D for plotting delete for final non plotting Kmeans
    
    #Transform the data
    df = pca.fit_transform(data)

    #Initialize the class object
    kmeans = KMeans(n_clusters= len(genrelist))
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    #Getting unique labels
    u_labels = np.unique(label)

    # plot graph  
    plot()

    pickle.dump(kmeans, open("save.pkl", "wb")) # save kmeans model

    # return Indices of Samples of one cluster
    return None

def defineClustNum(genre):
    # # assigning Genres to Clust numbers
#     if Genre == "Metal":    # Check Number of Clust Num Or change Clust Num to String
#         ClustNum = 1
#     elif Genre == "k-pop":
#         ClustNum = 2    
    clust_num = None
    return clust_num

def updateCluster(CSV_FILE_PATH = "../../data/Music_data/CSV_files"):
    with open(CSV_FILE_PATH, newline= ' ') as csvfile: # read csv file of  audio_features
        reader = csv.DictReader(csvfile)
        for csvdata in reader:
            audio_features.append(np.array(csvdata))

    clustering(audio_features)