#https://learn.datacamp.com/projects/449
#classify song genres from audio datasets

# %% import and read things
#imports pandas
import pandas as pd

#readin track metadata with genre labels
tracks = pd.read_csv("fma-rock-vs-hiphop.csv")

#read in track metrics with the features
echohonest_metrics = pd.read_json("echonest-metrics.json",precise_float= True)

#merge the relevant columns of tracks and echohonest_metrics on track ID retaining
#only track and genre from tracks
echo_tracks = pd.tracks.merge(left_on='track')
