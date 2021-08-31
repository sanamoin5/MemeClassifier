# rename politics files so that it can be used for training and testing
import os

path = 'C:/Users/Sana Moin/PycharmProjects/MemeClassifier/data/dataset_csvs'

for index, file in enumerate(os.listdir(path)):
    if os.path.isdir(os.path.join(path, file)) == True and file[0] != '.':
        new_path = os.path.join(path, file)
        classname = file
        for img_file in os.listdir(new_path):
            if img_file[0].isnumeric() == True and img_file.endswith('.jpg') or img_file.endswith(
                    '.jpeg') or img_file.endswith('.png'):
                os.rename(os.path.join(new_path, img_file), os.path.join(new_path, classname + '-' + img_file))
