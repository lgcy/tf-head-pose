# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:49:24 2018

@author: YZS
"""
import tensorflow as tf
import os, argparse
from tensorflow.contrib.slim import nets
import datasets
from PIL import Image

slim = tf.contrib.slim

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='D:/300W_LP', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='D:/300W_LP/300W_LP_filename_filtered.txt', type=str)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', help='Path to save the model.',
          default='./output', type=str)
    parser.add_argument('--image_size',dest = 'image_size', help = 'input image size', default = 224,type=int)

    args = parser.parse_args()
    
    return args

def test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    mae_yaw = 0
    mae_pitch = 0
    mae_roll = 0
    
    images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='image_batch')
    is_training = tf.placeholder(tf.bool)
    num_bins = 66
    keep_prob = 0
    
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope):
        net,endpoints = nets.resnet_v1.resnet_v1_50(images, num_classes = None, is_training = is_training)
    
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net,axis=[1,2])
        net = slim.dropout(net,keep_prob,scope='scope')
        yaw = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_yaw')
        pitch = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_pitch')
        roll = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_roll')
        
    yaw_predicted = tf.nn.softmax(yaw)
    pitch_predicted =tf.nn.softmax(pitch)
    roll_predicted = tf.nn.softmax(roll)
    
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = tf.convert_to_tensor(idx_tensor,tf.float32)
    
    yaw_predicted = tf.reduce_sum(yaw_predicted*idx_tensor,1)*3-99
    pitch_predicted = tf.reduce_sum(pitch_predicted*idx_tensor,1)*3-99
    roll_predicted = tf.reduce_sum(roll_predicted*idx_tensor,1)*3-99
   
    
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
        saver.restore(sess,ckpt)
        
        test_data = datasets.AFLW2000()
        for i in range(test_data.length):
            img = test_data[i][0]
            img = img.resize((224,224),Image.BILINEAR)
            precesssed_img = datasets.nomalizing(img,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            b_img = tf.expand_dims(precesssed_img,0)
            b_img = b_img.eval()
            
            feed_dict = {images: b_img,
                         is_training: False
                    }
            pre_yaw, pre_pitch, pre_roll = sess.run([yaw_predicted, pitch_predicted, roll_predicted],feed_dict=feed_dict)
            
            cont_labels = test_data[i][2]
            yaw_label = cont_labels[0]
            pitch_label = cont_labels[1]
            roll_label = cont_labels[2]
            
            mae_yaw = mae_yaw+ abs(pre_yaw-yaw_label)
            mae_pitch = mae_pitch+ abs(pre_pitch-pitch_label)
            mae_roll = mae_roll + abs(pre_roll-roll_label)
        
        mean_yaw = mae_yaw/test_data.length
        mean_pitch = mae_pitch/test_data.length
        mean_roll = mae_roll/test_data.length
            
    return mean_yaw, mean_pitch, mean_roll

def main(_):
    #args = parse_args()
    mean_yaw, mean_pitch, mean_roll = test()
    mae_all = (mean_pitch+mean_roll+mean_yaw)/3
    print("yaw_mae:", mean_yaw)
    print("pitch_mae:", mean_pitch)
    print("roll_mae:", mean_roll)
    print("mea_all", mae_all)
    
if __name__ =='__main__':
    tf.app.run()
    
            
            
            
        