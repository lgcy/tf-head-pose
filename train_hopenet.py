# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:23:37 2018

@author: YZS
"""
import tensorflow as tf
import os, argparse
from tensorflow.contrib.slim import nets
import datasets

slim = tf.contrib.slim

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=10, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='D:/300W_LP', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='D:/300W_LP/300W_LP_filename_filtered.txt', type=str)
    parser.add_argument('--pretrained_path', dest='pretrained_path', help='Path to put the pretrained model,like resetnet50.', 
          default = './resnet50/', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=1.0, type=float)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', help='Path to save the model.',
          default='./output', type=str)
    parser.add_argument('--image_size',dest = 'image_size', help = 'input image size', default = 224,type=int)
    parser.add_argument('--log_dir', dest='log_dir', help='log.',default='./output/log', type=str)

    args = parser.parse_args()
    print(args)
    return args


def main(_):
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    images = tf.placeholder(dtype=tf.float32, shape = [None,args.image_size,args.image_size,3],name = 'image_data')
    labels = tf.placeholder(dtype=tf.int32, shape = [None,3], name = 'cls_label')
    keep_prob = tf.placeholder(dtype=tf.float32, shape = [], name = 'keep_prob')
    cont_labels = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='cont_labels')
    is_training = tf.placeholder(tf.bool,name='is_training')
    num_bins = 66
    
    # Binned labels
    label_yaw = labels[:,0]
    label_pitch = labels[:,1]
    label_roll = labels[:,2]
    
    # Continuous labels
    label_yaw_cont = cont_labels[:,0]
    label_pitch_cont = cont_labels[:,1]
    label_roll_cont = cont_labels[:,2]
    
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net,endpoints = nets.resnet_v1.resnet_v1_50(images, num_classes = None, is_training = is_training)
    
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net,axis=[1,2])
        net = slim.dropout(net,keep_prob,scope='scope')
        yaw = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_yaw')
        pitch = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_pitch')
        roll = slim.fully_connected(net, num_outputs = num_bins, activation_fn=None, scope='fc_roll')
    
    # Cross entropy loss
    loss_yaw = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(label_yaw, tf.int64), logits=yaw)
    loss_pitch = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(label_pitch,tf.int64),logits=pitch)
    loss_roll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(label_roll,tf.int64),logits=roll)
    
    loss_yaw_ce = tf.reduce_mean(loss_yaw)
    loss_pitch_ce = tf.reduce_mean(loss_pitch)
    loss_roll_ce = tf.reduce_mean(loss_roll)
    
    # MSE loss
    yaw_predicted = tf.nn.softmax(yaw)
    pitch_predicted =tf.nn.softmax(pitch)
    roll_predicted = tf.nn.softmax(roll)
    
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = tf.convert_to_tensor(idx_tensor,dtype=tf.float32)
    
    yaw_predicted = tf.reduce_sum(yaw_predicted * idx_tensor, 1) * 3 - 99
    pitch_predicted = tf.reduce_sum(pitch_predicted * idx_tensor,1) * 3 - 99
    roll_predicted = tf.reduce_sum(roll_predicted * idx_tensor,1) * 3 - 99
    
    loss_reg_yaw = tf.reduce_mean(tf.square(yaw_predicted-label_yaw_cont))
    loss_reg_pitch = tf.reduce_mean(tf.square(pitch_predicted-label_pitch_cont))
    loss_reg_roll = tf.reduce_mean(tf.square(roll_predicted-label_roll_cont))

    # Total loss
    loss_yaw = tf.add_n([loss_yaw_ce,args.alpha * loss_reg_yaw])
    loss_pitch =tf.add_n([loss_pitch_ce, args.alpha * loss_reg_pitch])
    loss_roll = tf.add_n([loss_roll_ce,args.alpha * loss_reg_roll])
    
    loss_all = loss_yaw + loss_pitch + loss_roll
    
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(0.00001, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss_all, global_step=global_step)
    
    tf.summary.scalar('loss_yaw',loss_yaw)
    tf.summary.scalar('loss_pitch',loss_pitch)
    tf.summary.scalar('loss_roll',loss_roll)
    tf.summary.scalar('loss_all',loss_all)
    merged_summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    
    
    
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        if not os.path.exists(args.log_dir):
             os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        
        ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
        if (ckpt):
            tf.logging.info('restore the trained model')
            saver = tf.train.Saver(max_to_keep=5)
            saver.restore(sess,ckpt)
        else:
            tf.logging.info('load the pre-trained model')
            checkpoint_exclude_scopes = 'Logits'
            #exclusions = None
            
            if checkpoint_exclude_scopes:
                exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
                
            variables_to_restore = []
            for var in slim.get_model_variables():
                print(var)
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                    else:
                        variables_to_restore.append(var)
            
            saver_restore = tf.train.Saver(variables_to_restore)
            saver = tf.train.Saver(max_to_keep=5)
            saver_restore.restore(sess, os.path.join(args.pretrained_path,'resnet_v1_50.ckpt'))
        
        train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        dataset = datasets.Pose_300W_LP(data_dir='D:/300W_LP', filename_path=args.filename_list,
                                        batch_size=args.batch_size,
                                        image_size=args.image_size)
        
        for epoch in range(args.num_epochs):
            for i in range(dataset.length//args.batch_size):
                batch_images, batch_labels, batch_cont_labels = dataset.get()
                train_dict = {images: batch_images, 
                              labels: batch_labels,
                              is_training: True,
                              keep_prob: 0.5,
                              cont_labels: batch_cont_labels}
                _, loss, yaw_loss, pitch_loss, roll_loss, train_summary, step = sess.run([train_op,
                        loss_all, loss_yaw, loss_pitch,loss_roll, merged_summary_op, global_step],feed_dict = train_dict)
                train_writer.add_summary(train_summary,step)
                
                if step % 100==0:
                    tf.logging.info('the epoch %d: the loss of the step %d is: total_loss:%f, \
            loss_yaw:%f, loss_pitch:%f , loss_roll:%f'%(epoch, step, loss, yaw_loss, pitch_loss, roll_loss))
                
                if step % 500==0:
                    tf.logging.info('the epoch:%d, save the model for step %d'%(epoch,step))
                    saver.save(sess, os.path.join(args.checkpoint_dir,'model'), global_step=tf.cast(step*epoch, tf.int32))
                    
        tf.logging.info('==================Train Finished================')
                    
                
                
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
