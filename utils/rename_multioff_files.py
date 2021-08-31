# rename multioff files so that it can be used for training and testing
import pandas as pd
import os

path = 'C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/MultiOFF'

print('renaming files in folders')
for index, file in enumerate(os.listdir(path)):
    if os.path.isdir(os.path.join(path, file)) == True and file[0] != '.':
        new_path = os.path.join(path, file)
        classname = file
        if classname == 'Non-offensive':
            classname = 'Non_offensive'
        for img_file in os.listdir(new_path):
            if img_file.endswith('.jpg') or img_file.endswith(
                    '.jpeg') or img_file.endswith('.png'):
                os.rename(os.path.join(new_path, img_file), os.path.join(new_path, classname + '-' + img_file))

print('renaming files in csvs')
# rename file names in csvs
df_off = pd.read_csv('C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/MultiOFF/multioff_offensive.csv')
df_nonoff = pd.read_csv('C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/MultiOFF/multioff_non_offensive.csv')

df_off['image_name'] = 'offensive-' + df_off['image_name']
df_nonoff['image_name'] = 'Non_offensive-' + df_nonoff['image_name']

df_off.to_csv('C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/MultiOFF/multioff_offensive.csv')
df_nonoff.to_csv('C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/MultiOFF/multioff_non_offensive.csv')
