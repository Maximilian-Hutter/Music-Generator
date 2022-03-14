from data_preparation import getWaveFilesPath
from genre_clustering import updateCluster
import torchaudio
from torch.utils.data import Dataset
from genres import genrelist


class AudioDataset(Dataset):
    def __init__(self, chosen_genre):   # generate AI weights for each genre

        updateCluster() # update the kmeans cluster with new data
        wave_files_paths = getWaveFilesPath(chosen_genre)   # choose genre by generating AI weights for each Genre get the file paths from the kmeans cluster

        self.files = sorted(wave_files_paths)
    
    def __getitem__(self, index):   # get audio to dataloader
        
        waveform, sample_rate = torchaudio.load(self.files[index % len(self.files)])

        audio = {"waveform": waveform, "sample_rate": sample_rate}
        return audio 

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.files)