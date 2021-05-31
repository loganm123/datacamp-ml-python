#https://learn.datacamp.com/projects/449
#classify song genres from audio datasets

# %% import and read things
#imports pandas
import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv("\fma-rock-vs-hiphop")

# Read in track metrics with the features
echonest_metrics = pd.read_json("C:\GitHub\datacamp-ml-python\scikit practice\echonest-metrics.json")

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = pd.merge(left = echonest_metrics, right = tracks[['track_id','genre_top']], on = 'track_id', how = 'left')

# Inspect the resultant dataframe
echo_tracks.head(10)
