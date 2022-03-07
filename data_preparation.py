from clustering import Clustering



data # data split into song and metadata (or metadata seperate csv file while song seperate wav file)

Samples = []

Indices = Clustering(ChosenGenre, metadata) # metadata is a list of arrays and has 9 dimensions

for i in Indices:
    Samples.append(data[i])