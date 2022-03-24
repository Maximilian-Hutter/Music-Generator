from get_path_by_genre import getWaveFilesPath
from genre_clustering import updateCluster
import torchaudio
from torch.utils.data import Dataset
from transform_data import transform_to_label

class AudioDataset(Dataset):
    def __init__(self, chosen_genre):   # generate AI weights for each genre

        updateCluster() # update the kmeans cluster with new data
        wave_files_paths = getWaveFilesPath(chosen_genre)   # choose genre by generating AI weights for each Genre get the file paths from the kmeans cluster

        self.files = sorted(wave_files_paths)
    
    def __getitem__(self, index):   # get audio to dataloader
        
        wave_x, sample_rate_x = torchaudio.load(self.files[index % len(self.files)])

        # transform the input values to the labels 
        wave_y = transform_to_label(wave_x)   # if not possible use different dataset
        sample_rate_y = sample_rate_x

        audio = {"wave_x": wave_x, "sample_rate_x": sample_rate_x, "wave_y": wave_y, "sample_rate_y": sample_rate_y}

        return audio

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.files)