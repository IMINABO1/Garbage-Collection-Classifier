from time import sleep
import numpy as np
import pandas as pd 
import os

import keras.applications.xception as xception
import tensorflow as tf
import re
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

#print("hey")

#constants
IMAGE_WIDTH = 320    
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

base_path = "../data_Set/archive/garbage_classification/"

# Dictionary to save our 13 classes
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological', 12:'electronics'}


# creating dataset and image data generator
# Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
    return df

# list conatining all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, note that each folder of the dataset has one class of data
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])
    
    filenames_list = filenames_list  +filenames
    categories_list = categories_list + [category] * len(filenames)
    
df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

#print('number of elements = ' , len(df))

#Change the categories from numbers to names
df["category"] = df["category"].replace(categories) 

# We first split the data into two sets and then split the validate_df to two sets
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

test_df = test_df.reset_index(drop=True)


#test  data gen
test_datagen = image.ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False 
)

#load the model
test_model= tf.keras.models.load_model("model")

#image processing
def image_processing(image_input):
    image_input = load_img(image_input, target_size=(320, 320))
    image_input = img_to_array(image_input)
    image_input = np.array(image_input)
    image_input = np.expand_dims(image_input, axis=0)

    return image_input

## prediction

'''
gen_label_map={0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'electronics', 6: 'green-glass', 7: 'metal', 8: 'paper', 9: 'plastic', 10: 'shoes', 11: 'trash', 12: 'white-glass'}
'''

def make_predict(procesed_image,tst_gen=test_generator,tst_model=test_model):
    a_prediction = tst_model.predict(procesed_image)
    gen_label_map = tst_gen.class_indices
    gen_label_map = dict((v,k) for k,v in gen_label_map.items())
    print(gen_label_map)
    preds = a_prediction.argmax(1)
    #preds = [gen_label_map[item] for item in preds]

    return preds.tolist()

#the_image="../data_Set/archive/my_test/v.png"



def main(img):
    print("starting")
    the_image=image_processing(img)
    print("img accepted")
    a=make_predict(the_image,test_generator,test_model)
    print("prediction made")
    print(a)
    #sleep(5)
    return a
    
#main()
'''

if __name__== '__main__':
    main()
'''