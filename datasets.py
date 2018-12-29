import os
import numpy as np
from random import randint
import tensorflow as tf
from PIL import Image, ImageFilter
import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def rescale(image):
    w = image.size[0]
    h = image.size[1]
    #resize to 240
    outsize = 240
    
    if w < h:
        return image.resize((outsize,round(h/w * outsize)),Image.BILINEAR)
    else:
        return image.resize((round(w/h * outsize),outsize),Image.BILINEAR)

def random_crop(image):
    w = image.size[0]
    h = image.size[1]
    size =224
    new_left = randint(0,w - size)
    new_upper = randint(0,h - size)
    return image.crop((new_left,new_upper,size+new_left,size+new_upper))

def nomalizing(image,mean_value,std):
    image = np.array(image)
    image = image/255.0
    for i in range(3):
        image[:,:,i] = (image[:,:,i] - mean_value[i])/std[i]
    return image


class Pose_300W_LP():
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, batch_size,image_size,img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        #self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.batch_size=batch_size
        self.image_size=image_size

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.cursor=0
        #self.batch_size=16#args.batch_size


    def get(self):

        images = np.zeros((self.batch_size,self.image_size, self.image_size, 3))
        Llabels = np.zeros((self.batch_size,3),np.int32)
        Lcont_labels=np.zeros((self.batch_size,3))
        count=0
        
        while count<self.batch_size:
            img = Image.open(os.path.join(self.data_dir, self.X_train[self.cursor] + self.img_ext))
            #print('img', img.shape)
            img = img.convert(self.image_mode)
            mat_path = os.path.join(self.data_dir, self.y_train[self.cursor] + self.annot_ext)

            # Crop the face loosely
            pt2d = utils.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0, :])
            y_min = min(pt2d[1, :])
            x_max = max(pt2d[0, :])
            y_max = max(pt2d[1, :])
            # k = 0.2 to 0.40
            k = np.random.random_sample() * 0.2 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            
            
            # We get the pose in radians
            pose = utils.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)
                
            #preprocess
            img = rescale(img)
            img = random_crop(img)
            img = nomalizing(img,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            # Bin values
            bins = np.array(range(-99, 102, 3))
            binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

            labels = binned_pose

            cont_labels=[float(yaw),float(pitch),float(roll)]


            images[count, :, :, :] = img
            Llabels[count]=labels
            Lcont_labels[count]=cont_labels


            count+=1
            self.cursor+=1
            if self.cursor >= len(self.X_train):
                np.random.shuffle(self.X_train)
                self.cursor = 0
                print("self.cursor ====0")
                
            #print(self.X_train[0])

        return images,Llabels,Lcont_labels

class AFLW2000():
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = np.digitize([yaw, pitch, roll], bins) - 1
        cont_labels = [yaw, pitch, roll]

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length

#if __name__ =='__main__':
#    data_dir = 'D:/300W_LP'
#    filename_path = 'D:/300W_LP/300W_LP_filename_filtered.txt'
#    transform = None
#    data = Pose_300W_LP(data_dir,filename_path,1,224)


