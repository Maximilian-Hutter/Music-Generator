from tinytag import TinyTag
from clustering import Clustering

data # data split into song and metadata

Samples = []

Indices = Clustering(ChosenGenre, metadata)

for i in Indices:
    Samples.append(data[i])