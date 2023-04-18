# load dependencies

# misc
import datetime
import os, sys, shutil

# basics
import numpy as np
from numpy import loadtxt
import pandas as pd

from PIL import Image


# keras


# set file path variables
train_path = 'C://Users//hp//Desktop//Project-1//train_images'
test_path = 'C://Users//hp//Desktop//Project-1//test_images'

# load csv files with image file names and labels as pandas dataframes
train_data = pd.read_csv('C:/Users/hp/Desktop/Project-1/train.csv')
test_data = pd.read_csv('C:/Users/hp/Desktop/Project-1/test.csv')



# look at training data
train_data.head()

# look at test data
test_data.head()

# number of images in test & train data
print('Number of images in training set is {}'.format(len(train_data)))
print('Number of images in test set is {}'.format(len(test_data)))


# store the class information in some variables for convenience
class_labels = [0,1,2,3,4]
class_dict = {0:'No Glaucoma', 1:'Mild Glaucoma', 2:'Moderate Glaucoma', 3:'Severe Glaucoma', 4:'Proliferative Glaucoma'}
class_list = ['No Glaucoma', 'Mild Glaucoma', 'Moderate Glaucoma', 'Severe Glaucoma', 'Proliferative Glaucoma']


# look at the distribution of the training data into the 5 classes
train_data.diagnosis.value_counts()


# function to determine largest and smallest dimensions of the images
def get_dimensions(df, path):
    max_width = 0
    max_height = 0
    min_width = 0
    max_height = 0
    
    file_names = df['id_code']
    
    for index, file_name in enumerate(file_names):
        current_image = Image.open(path+'//'+file_name+'.png')

        width, height = current_image.size
        
        # set initial values
        if max_width == 0:
            max_width = width
            min_width = width
            max_height = height
            min_height = height
        
        if width > max_width:
            max_width = width
        if width < min_width:
            min_width = width
            
        if height > max_height:
            max_height = height
        if height < min_height:
            min_height = height
    
    print('Minimum width: {},'.format(min_width))
    print('Maximum width: {},'.format(max_width))
    print('************')
    print('Minimum height: {}'.format(min_height))
    print('Maximum height: {}'.format(max_height))
    


# look at the train images sizes
get_dimensions(train_data, train_path)



# look at the test images sizes
get_dimensions(test_data, test_path)



old_train = train_path
new_folder = 'data_organized'

dir_names = ['train', 'val', 'test']

os.mkdir(new_folder)

for d in dir_names:
    new_dir = os.path.join(new_folder, d)
    os.mkdir(new_dir)


for label in class_labels:
    print('Moving Class {} images.'.format(label))
    for d in dir_names:
        new_dir = os.path.join(new_folder, d, str(label))
        os.mkdir(new_dir)
        print('created \t')
    temp = train_data[train_data.diagnosis == label]
    train, validate, test = np.split(temp.sample(frac=1), [int(.8*len(temp)), int(.9*len(temp))])
    print('Split {} imgs into {} train, {} val, and {} test examples.'.format(len(temp),
                                                                               len(train),
                                                                               len(validate),
                                                                               len(test)))
    for i, temp in enumerate([train, validate, test]):
        for row in temp.index:
            filename = temp['id_code'][row] + '.png'
            origin = os.path.join(old_train + '/' + filename)
            destination = os.path.join(new_folder + '/' + dir_names[i] + '/' + str(label) + '/' + filename)
            
            shutil.copy(origin, destination)




