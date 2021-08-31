""" Creates a dataset by reading image links obtained from google images through 'Image Link Grabber' plugin and
 downloads each image for those links. Images are stored in data directory under their respective class folder.
"""
from fastai.vision.utils import download_images
from fastai.vision import *
from fastai.imports import *
import os
import zipfile
import time


# downloads images to specific directories in data folder for each class
def download_images_to_folder(file, folder):
    path = Path('../data/dataset_csvs/')
    dest = path / folder
    dest.mkdir(parents=True, exist_ok=True)
    download_images(path / file, dest)


# used if running on google colab, to zip the folder for downloading dataset
def zip_folder(data_class_name):
    zip_file = zipfile.ZipFile(data_class_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    filepath = 'data/dataset_csvs/' + data_class_name + '/'
    for root, dirs, files in os.walk(filepath):
        for file in files:
            zip_file.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), os.path.join(filepath, '../..')))
    zip_file.close()


# opens all csv files in data folder and downloads the images to their respective class folders
def main():
    for all_files in os.listdir('../data/dataset_csvs/'):
        if all_files.endswith('.csv'):
            filename = all_files
            class_name = filename.split('.')[0]
            print('Downloading files for {0} class'.format(class_name))
            print(filename)
            download_images_to_folder(filename, class_name)
            print('... Downloaded files for {0} class'.format(class_name))
            time.sleep(5)
            # uncomment the line below if folder needs to be zipped
            # zip_folder(class_name)


if __name__ == "__main__":
    main()
