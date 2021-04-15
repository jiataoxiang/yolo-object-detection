import os
import shutil
import random
import numpy as np
from PIL import Image, ImageOps


source_label_path = './train/labels/'
source_image_path = './train/images/'
destination_label_path = './test/labels/'
destination_image_path = './test/images/'
# ratio kept for each axis
keep_rates = [0.9,
              0.8,
              0.7]


label_file_names = [f for f in os.listdir(source_label_path) if f[-4:] == '.txt']

def flip_label(label):
    flipped = label.copy()
    flipped[:,1] = 1-flipped[:,1]
    return flipped

def crop_label(label, keep_rate):
    cropped = label.copy()
    cropped[:,1:3] = cropped[:,1:3]/keep_rate - (1-keep_rate)/2/keep_rate
    cropped[:,3:] /= keep_rate
    
    cropped = np.delete(cropped, np.where(cropped[:,1] - cropped[:,3] / 2 <= 0), axis=0)
    cropped = np.delete(cropped, np.where(cropped[:,1] + cropped[:,3] / 2 >= 1), axis=0)
    cropped = np.delete(cropped, np.where(cropped[:,2] - cropped[:,4] / 2 <= 0), axis=0)
    cropped = np.delete(cropped, np.where(cropped[:,2] + cropped[:,4] / 2 >= 1), axis=0)
    return cropped

def aug_label():
    for file_name in label_file_names:
        label = np.loadtxt(source_label_path + file_name)
        label = label.reshape((-1, 5))
        flipped = flip_label(label)
        np.savetxt(source_label_path+file_name[:-4]+"_flipped.txt", flipped, fmt="%1.18f")
    
        for keep_rate in keep_rates:
            cropped = crop_label(label, keep_rate)
            np.savetxt(source_label_path+file_name[:-4]+"_cropped"+str(100-int(keep_rate*100))+".txt", cropped, fmt="%1.18f")
            
            flipped = flip_label(cropped)
            np.savetxt(source_label_path+file_name[:-4]+"_cropped"+str(100-int(keep_rate*100))+"_flipped.txt", flipped, fmt="%1.18f")


image_file_names = [f for f in os.listdir(source_image_path) if f[-4:] == '.jpg']

def aug_image():
    for file_name in image_file_names:
        image = Image.open(source_image_path + file_name)
        flipped = ImageOps.mirror(image)
        flipped.save(source_image_path+file_name[:-4]+"_flipped.jpg")
    
        for keep_rate in keep_rates:
            left = round(image.size[0] * (1-keep_rate)/2)
            top = round(image.size[1] * (1-keep_rate)/2)
            right = image.size[0] - left
            bottom = image.size[1] - top
            cropped = image.crop((left, top, right, bottom))
            cropped.save(source_image_path+file_name[:-4]+"_cropped"+str(100-int(keep_rate*100))+".jpg")
            
            flipped = ImageOps.mirror(cropped)
            flipped.save(source_image_path+file_name[:-4]+"_cropped"+str(100-int(keep_rate*100))+"_flipped.jpg")


# keep train_ratio of the data in the source folders
# move 1-train_ratio of the data to the destination folders
def split_data(train_ratio=0.8):
    file_names = [f[:-4] for f in os.listdir(source_label_path) if f[-4:] == '.txt']
    random.shuffle(file_names)
    split_index = int(len(file_names) * train_ratio)
    
    test_file_names = file_names[split_index:]
    for file_name in test_file_names:
        shutil.move(source_label_path+file_name+'.txt', destination_label_path+file_name+'.txt')
        shutil.move(source_image_path+file_name+'.jpg', destination_image_path+file_name+'.jpg')
        

if __name__ == "__main__":
    aug_label()
    aug_image()
    split_data(train_ratio=0.8)
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    