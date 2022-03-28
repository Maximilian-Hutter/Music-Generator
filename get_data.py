from get_path_by_genre import getWaveFilesPath
from genre_clustering import updateCluster
import torchaudio
from torch.utils.data import Dataset
import os

class AudioDataset(Dataset):
    def __init__(self,WAV_DIRECTORY_PATH,LABEL_DIRECTORY_PATH, chosen_genre):   # generate AI weights for each genre

        if chosen_genre == None:
            wave_files_paths = os.listdir(WAV_DIRECTORY_PATH)
        else:
            updateCluster() # update the kmeans cluster with new data
            wave_files_paths = getWaveFilesPath(chosen_genre)   # choose genre by generating AI weights for each Genre get the file paths from the kmeans cluster

        self.files = sorted(wave_files_paths)
        self.files = sorted(os.listdir(LABEL_DIRECTORY_PATH))
        
    
    def __getitem__(self, index):   # get audio to dataloader
        
        wave_x, sample_rate_x = torchaudio.load(self.files[index % len(self.files)])
        wave_y, sample_rate_y = torchaudio.load(self.files[index % len(self.labels)])

        # transform the input values to the labels 

        audio = {"wave_x": wave_x, "sample_rate_x": sample_rate_x, "wave_y": wave_y, "sample_rate_y": sample_rate_y}

        return audio

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.files)