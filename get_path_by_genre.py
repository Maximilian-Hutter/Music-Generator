from genre_clustering import defineClustNum
import numpy as np
import pickle
import os

WAV_DIRECTORY_PATH = "../../data/Music_data/WAV_files"  # path to wav audio files

def  getIndices(chosen_genre): # get list indices from kmeans clusters

    def clusterIndices(clust_num, labels_array): # get index from all samples in one cluster
        return np.where(labels_array == clust_num)[0]

    clust_num = defineClustNum(chosen_genre)
    kmeans = pickle.load(open("save.plk", "rb"))

    indices = clusterIndices(clust_num, kmeans.labels_)  # Need to assign genres to Clust numbers, Clust Num = "Genre", create tabelle für ClustNum and Genre
    
    return indices

def getWaveFilesPath(chosen_genre):  # get wavfile paths using indices from clusters and os list directory

    wave_file_paths = []

    indices = getIndices(chosen_genre)
    all_wave_files = os.listdir(WAV_DIRECTORY_PATH) # if data gotten is inconsistent sort before getting wave paths
    for i in indices:
        wave_files = all_wave_files[i]
        wave_file_path = WAV_DIRECTORY_PATH + "/" + wave_files
        wave_file_paths.append(wave_file_path)
    return wave_file_paths