# reads and re-formats MultiOFF dataset files to suit the efficientnet classifier dataset folder structure requirement
import os
import pandas as pd
import shutil

# read all files
df_train = pd.read_csv(
    '..\data\MultiOFF_Dataset-20210822T105344Z-001\MultiOFF_Dataset\Split Dataset\Training_meme_dataset.csv')
df_test = pd.read_csv(
    '..\data\MultiOFF_Dataset-20210822T105344Z-001\MultiOFF_Dataset\Split Dataset\Testing_meme_dataset.csv')
df_val = pd.read_csv(
    '..\data\MultiOFF_Dataset-20210822T105344Z-001\MultiOFF_Dataset\Split Dataset\Validation_meme_dataset.csv')

# combine all files in 1 df
df_all = df_train.append(df_test).append(df_val)

print(df_all.label.value_counts())

# create new directories to store images, one folder for each class
os.mkdir('..\data\MultiOFF')
os.mkdir('..\data\MultiOFF\\offensive')
os.mkdir('..\data\MultiOFF\\Non-offensive')

# separate offensive and non offensive dataframes
df_offensive = df_all[df_all['label'] == 'offensive']
df_non_offensive = df_all[df_all['label'] == 'Non-offensiv']

# provide path to current and new dir
path = '..\data\MultiOFF_Dataset-20210822T105344Z-001\MultiOFF_Dataset\Labelled Images'
new_off_path = '..\data\MultiOFF\\offensive'
new_nonoff_path = '..\data\MultiOFF\\non-offensive'

# copy images of offensive category to new folder
for index, row in df_offensive.iterrows():
    shutil.copy(os.path.join(path, row['image_name']), new_off_path)

# copy images of non-offensive category to new folder
for index, row in df_non_offensive.iterrows():
    shutil.copy(os.path.join(path, row['image_name']), new_nonoff_path)

# update label tag for non-offensive
df_non_offensive['label'] = df_non_offensive['label'].map({'Non-offensiv': 'non-offensive'})

# save offensive and non-offensive df into new csv files
df_offensive.to_csv('..\data\MultiOFF\\multioff_offensive.csv')
df_non_offensive.to_csv('..\data\MultiOFF\multioff_non_offensive.csv')
