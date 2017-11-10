import numpy as np
import tensorflow as tf
import cv2
from time import gmtime, strftime
from random import shuffle
import random

model_name = 'model64_4l_2017-11-080843_re14'

# Parameters
learning_rate = 0.0001
epochs = 2500
batch_size = 50

#Training dataset loading parameters
fisrt_file_index = 1
end_file_index = 13

fisrt_file_index_vot = 1
end_file_index_vot = 13
frame_step_vot = 2
# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network Parameters
n_output = 4  # Cx,Cy,W,H
dropout = 0.5  # Dropout, probability to keep units

#Training resize image size
width_resize = 64
height_resize = 64

#weight and bias initial parameter
mu = 0
sigma = 0.02

#Crop image parameters
k1=3

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], mean = mu, stddev = sigma)),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], mean = mu, stddev = sigma)),
    'wd1': tf.Variable(tf.random_normal([(width_resize/4)**2*64, 1024*4], mean = mu, stddev = sigma)),

    'wc_p1': tf.Variable(tf.random_normal([5, 5, 3, 32], mean = mu, stddev = sigma)),
    'wc_p2': tf.Variable(tf.random_normal([5, 5, 32, 64], mean = mu, stddev = sigma)),
    'wd_p1': tf.Variable(tf.random_normal([(width_resize/4)**2*64, 1024*4], mean = mu, stddev = sigma)),

    'wd_y_pre1': tf.Variable(tf.random_normal([4, 1024*4], mean = mu, stddev = sigma)),

    'wd2': tf.Variable(tf.random_normal([1024*4, 1024*4], mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.random_normal([1024*4, n_output], mean = mu, stddev = sigma))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32], mean = mu, stddev = sigma)),
    'bc2': tf.Variable(tf.random_normal([64], mean = mu, stddev = sigma)),

    'bc_p1': tf.Variable(tf.random_normal([32], mean = mu, stddev = sigma)),
    'bc_p2': tf.Variable(tf.random_normal([64], mean = mu, stddev = sigma)),

    'bd1': tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma)),
    'bd2': tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.random_normal([n_output], mean = mu, stddev = sigma))}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def conv_net(x_cur,x_pre, x_y_pre, weights, biases, dropout):

    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x_cur, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024*4
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#######################################################################

    # Layer 1 - 28*28*1 to 14*14*32
    conv_p1 = conv2d(x_pre, weights['wc_p1'], biases['bc_p1'])
    conv_p1 = maxpool2d(conv_p1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv_p2 = conv2d(conv_p1, weights['wc_p2'], biases['bc_p2'])
    conv_p2 = maxpool2d(conv_p2, k=2)

    # Fully connected layer - 7*7*64 to 1024*4
    fc_p1 = tf.reshape(conv_p2, [-1, weights['wd_p1'].get_shape().as_list()[0]])
#######################################################################

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc_p1 = tf.matmul(fc_p1, weights['wd_p1'])
    #add previous output to full connect layer
    fc_p_y_pre1 = tf.matmul(x_y_pre, weights['wd_y_pre1'])

    fc_d1_temp = tf.add(fc1, fc_p1)
    fc_d1 = tf.add(fc_d1_temp, fc_p_y_pre1)

    fc_d1 = tf.nn.relu(fc_d1)
    fc_d1 = tf.nn.dropout(fc_d1, dropout)

    # Fully connected layer - 7*7*64 to 1024*4
    fc_d2 = tf.add(tf.matmul(fc_d1, weights['wd2']), biases['bd2'])
    fc_d2 = tf.nn.relu(fc_d2)
    fc_d2 = tf.nn.dropout(fc_d2, dropout)

    # Output Layer - class prediction - 1024 to 4
    out = tf.add(tf.matmul(fc_d2, weights['out']), biases['out'])
    return out

# tf Graph input
x_cur = tf.placeholder(tf.float32, [None, width_resize, height_resize, 3])
x_pre = tf.placeholder(tf.float32, [None, width_resize, height_resize, 3])
x_y_pre = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x_cur, x_pre, x_y_pre, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(logits-y))
cost_L1 = tf.reduce_mean(tf.square(logits-y)/2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# correct_pred_max = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# correct_pred_min = tf.equal(tf.argmin(logits, 1), tf.argmin(y, 1))
# correct_pred = tf.equal(correct_pred_max, correct_pred_min)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#Create a saver object which will save all the variables
saver = tf.train.Saver()
#######################################################################

#add one more dimension
x_trainset_cur = [] # init numpy array
x_trainset_pre = [] #
x_trainset_y_pre = []
y_trainset_cur = [] # init numpy array

x_trainset_y_pre_shuf_noise = []
y_trainset_cur_shuf_noise = [] # init numpy array
#######################################################################
for file_num in range(fisrt_file_index, end_file_index):
    # file_num = 2
    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_position_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/

    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len

    for num_img in range(1, frame_len):
        y_temp_cur = groundtruth_frame[num_img]
        y_temp_cur = y_temp_cur.split()
        y_temp_float_cur = [float(i) for i in y_temp_cur]

        y_temp_pre = groundtruth_frame[num_img-1]
        y_temp_pre = y_temp_pre.split()
        y_temp_float_pre = [float(i) for i in y_temp_pre]

        frame_index_cur = int(y_temp_float_cur[0])
        frame_index_pre = int(y_temp_float_pre[0])

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        # image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_path_cur = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        # image_path_cur = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_cur = cv2.imread(image_path_cur)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)

        height_original, width_original = image_read_cur.shape[:2]
        print 'width_original', width_original, ' height_original', height_original

        #Use previous frame to crop previous frame and current frame, and use current y as turth value
        x_value_pre = [y_temp_float_pre[1],y_temp_float_pre[3],y_temp_float_pre[5],y_temp_float_pre[7]]
        y_value_pre = [y_temp_float_pre[2],y_temp_float_pre[4],y_temp_float_pre[6],y_temp_float_pre[8]]
        x_value_cur = [y_temp_float_cur[1],y_temp_float_cur[3],y_temp_float_cur[5],y_temp_float_cur[7]]
        y_value_cur = [y_temp_float_cur[2],y_temp_float_cur[4],y_temp_float_cur[6],y_temp_float_cur[8]]

        height = int(max(y_value_pre)-min(y_value_pre))
        width = int(max(x_value_pre)-min(x_value_pre))
        x_center = int((max(x_value_pre)+min(x_value_pre))/2.0)
        y_center = int((max(y_value_pre)+min(y_value_pre))/2.0)

        print 'num_img',num_img,' x_c:', x_center,' y_c:',y_center, 'w:',width,' h:',height
        print 'y width_crop', width, ' y height_crop', height

        y_crop_start = y_center - int(height*k1/2)
        y_crop_end = y_center + int(height*k1/2)
        x_crop_start = x_center - int(width*k1/2)
        x_crop_end = x_center + int(width*k1/2)

        if y_crop_start < 0:
            y_crop_start = 0
        if y_crop_end < 0:
            y_crop_end = 0
        if x_crop_start < 0:
            x_crop_start = 0
        if x_crop_end < 0:
            x_crop_end = 0

        k1crop_img_cur = image_read_cur[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        k1crop_img_pre = image_read_pre[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

        height_crop, width_crop = k1crop_img_cur.shape[:2]
        print 'width_crop', width_crop, ' height_crop', height_crop

        image_read_resize_cur = cv2.resize(k1crop_img_cur, (width_resize, height_resize))
        image_train_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_train_pre = image_read_resize_pre/255.0-.5

        #use current y as turth value
        x_bl_cur = (min(x_value_cur))#x bottom left
        y_bl_cur = (min(y_value_cur))#y bottom left
        x_tr_cur = (max(x_value_cur))#x top right
        y_tr_cur = (max(y_value_cur))#y top right

        #project original location project to croped resize box location
        x_bl_box_cur = (x_bl_cur-x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_bl_box_cur = (y_bl_cur-y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5
        x_tr_box_cur = (x_tr_cur-x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_tr_box_cur = (y_tr_cur-y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5

        y_train_cur = [x_bl_box_cur, y_bl_box_cur, x_tr_box_cur, y_tr_box_cur]

        #use current y as turth value
        x_bl_pre = (min(x_value_pre))#x bottom left
        y_bl_pre = (min(y_value_pre))#y bottom left
        x_tr_pre = (max(x_value_pre))#x top right
        y_tr_pre = (max(y_value_pre))#y top right

        #project original location project to croped resize box location
        x_bl_box_pre = (x_bl_pre - x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_bl_box_pre = (y_bl_pre - y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5
        x_tr_box_pre = (x_tr_pre - x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_tr_box_pre = (y_tr_pre - y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5

        y_train_pre = [x_bl_box_pre, y_bl_box_pre, x_tr_box_pre, y_tr_box_pre]


        # img_rec_out_cur = cv2.rectangle(image_train_cur,(int(y_train_cur[0]),int(y_train_cur[1])),(int(y_train_cur[2]),int(y_train_cur[3])),(0,255,0),1)
        # img_rec_out_pre = cv2.rectangle(image_train_pre,(int(y_train_pre[0]),int(y_train_pre[1])),(int(y_train_pre[2]),int(y_train_pre[3])),(0,255,0),1)

        # image_train_cur = (img_rec_out_cur+0.5)*255.0
        # image_train_pre = (img_rec_out_pre+0.5)*255.0
        # vis = np.concatenate((image_train_cur, image_train_pre), axis=1)
        # cv2.imwrite('./image_result/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

        x_trainset_cur.append(image_train_cur)
        x_trainset_pre.append(image_train_pre)
        x_trainset_y_pre.append(y_train_pre)
        y_trainset_cur.append(y_train_cur)

        x_trainset_y_pre_shuf_noise.append(y_train_pre)
        y_trainset_cur_shuf_noise.append(y_train_cur)

        # image_train_cur = (image_train_cur+0.5)*255.0
        # image_train_pre = (image_train_pre+0.5)*255.0
        # vis = np.concatenate((image_train_cur, image_train_pre), axis=1)
        # cv2.imwrite('./image_result/image_train_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

#######################################################################

#For vot video dataset load
#######################################################################
for file_num in range(fisrt_file_index_vot, end_file_index_vot):

    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/vot_video000"+format(file_num, '02d')+"/groundtruth.txt", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_position_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/

    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len

    for num_img in range(6, frame_len, frame_step_vot):
        print 'num_img: ', num_img

        y_temp_cur = groundtruth_frame[num_img-1]
        y_temp_cur = y_temp_cur.split(",")
        y_temp_float_cur = np.array(y_temp_cur)
        y_temp_float_cur = y_temp_float_cur.astype(np.float)

        y_temp_pre = groundtruth_frame[num_img-1-frame_step_vot]
        y_temp_pre = y_temp_pre.split(",")
        y_temp_float_pre = np.array(y_temp_pre)
        y_temp_float_pre = y_temp_float_pre.astype(np.float)

        frame_index_cur = num_img
        frame_index_pre = num_img - frame_step_vot

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        # image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_path_cur = './Track_human_dataset/vot_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        # image_path_cur = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_cur = cv2.imread(image_path_cur)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/vot_video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)

        height_original, width_original = image_read_cur.shape[:2]
        print 'width_original', width_original, ' height_original', height_original

        #Use previous frame to crop previous frame and current frame, and use current y as turth value
        x_value_pre = [y_temp_float_pre[0],y_temp_float_pre[2],y_temp_float_pre[4],y_temp_float_pre[6]]
        y_value_pre = [y_temp_float_pre[1],y_temp_float_pre[3],y_temp_float_pre[5],y_temp_float_pre[7]]
        x_value_cur = [y_temp_float_cur[0],y_temp_float_cur[2],y_temp_float_cur[4],y_temp_float_cur[6]]
        y_value_cur = [y_temp_float_cur[1],y_temp_float_cur[3],y_temp_float_cur[5],y_temp_float_cur[7]]

        height = int(max(y_value_pre)-min(y_value_pre))
        width = int(max(x_value_pre)-min(x_value_pre))
        x_center = int((max(x_value_pre)+min(x_value_pre))/2.0)
        y_center = int((max(y_value_pre)+min(y_value_pre))/2.0)

        print 'num_img',num_img,' x_c:', x_center,' y_c:',y_center, 'w:',width,' h:',height
        print 'y width_crop', width, ' y height_crop', height

        y_crop_start = y_center - int(height*k1/2)
        y_crop_end = y_center + int(height*k1/2)
        x_crop_start = x_center - int(width*k1/2)
        x_crop_end = x_center + int(width*k1/2)

        if y_crop_start < 0:
            y_crop_start = 0
        if y_crop_end < 0:
            y_crop_end = 0
        if x_crop_start < 0:
            x_crop_start = 0
        if x_crop_end < 0:
            x_crop_end = 0

        k1crop_img_cur = image_read_cur[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        k1crop_img_pre = image_read_pre[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

        height_crop, width_crop = k1crop_img_cur.shape[:2]
        print 'width_crop', width_crop, ' height_crop', height_crop

        image_read_resize_cur = cv2.resize(k1crop_img_cur, (width_resize, height_resize))
        image_train_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_train_pre = image_read_resize_pre/255.0-.5

        #use current y as turth value
        x_bl_cur = (min(x_value_cur))#x bottom left
        y_bl_cur = (min(y_value_cur))#y bottom left
        x_tr_cur = (max(x_value_cur))#x top right
        y_tr_cur = (max(y_value_cur))#y top right

        #project original location project to croped resize box location
        x_bl_box_cur = (x_bl_cur-x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_bl_box_cur = (y_bl_cur-y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5
        x_tr_box_cur = (x_tr_cur-x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_tr_box_cur = (y_tr_cur-y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5

        y_train_cur = [x_bl_box_cur, y_bl_box_cur, x_tr_box_cur, y_tr_box_cur]

        #use current y as turth value
        x_bl_pre = (min(x_value_pre))#x bottom left
        y_bl_pre = (min(y_value_pre))#y bottom left
        x_tr_pre = (max(x_value_pre))#x top right
        y_tr_pre = (max(y_value_pre))#y top right

        #project original location project to croped resize box location
        x_bl_box_pre = (x_bl_pre - x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_bl_box_pre = (y_bl_pre - y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5
        x_tr_box_pre = (x_tr_pre - x_crop_start)*(float(width_resize)/float(width*k1))/width_resize-0.5
        y_tr_box_pre = (y_tr_pre - y_crop_start)*(float(height_resize)/float(height*k1))/height_resize-0.5

        y_train_pre = [x_bl_box_pre, y_bl_box_pre, x_tr_box_pre, y_tr_box_pre]

        # img_rec_out_cur = cv2.rectangle(image_train_cur,(int(y_train_cur[0]),int(y_train_cur[1])),(int(y_train_cur[2]),int(y_train_cur[3])),(0,255,0),1)
        # img_rec_out_pre = cv2.rectangle(image_train_pre,(int(y_train_pre[0]),int(y_train_pre[1])),(int(y_train_pre[2]),int(y_train_pre[3])),(0,255,0),1)
        #
        # image_train_cur = (img_rec_out_cur+0.5)*255.0
        # image_train_pre = (img_rec_out_pre+0.5)*255.0
        # vis = np.concatenate((image_train_cur, image_train_pre), axis=1)
        # cv2.imwrite('./image_result/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

        # img_rec_out_origin_cur = cv2.rectangle(image_read_cur,(int(x_bl_cur),int(y_bl_cur)),(int(x_tr_cur),int(y_tr_cur)),(0,255,0),2)
        # image_test_orgin_cur = img_rec_out_origin_cur
        # cv2.imwrite('./image_result/image_vot_origin_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', image_test_orgin_cur)

        x_trainset_cur.append(image_train_cur)
        x_trainset_pre.append(image_train_pre)
        x_trainset_y_pre.append(y_train_pre)
        y_trainset_cur.append(y_train_cur)

        # x_trainset_y_pre_shuf_noise.append(y_train_pre)
        # y_trainset_cur_shuf_noise.append(y_train_cur)

        # image_train_cur = (image_train_cur+0.5)*255.0
        # image_train_pre = (image_train_pre+0.5)*255.0
        # vis = np.concatenate((image_train_cur, image_train_pre), axis=1)
        # cv2.imwrite('./image_result/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

print 'x_trainset_cur length', len(x_trainset_cur)
print 'x_trainset_pre length', len(x_trainset_pre)
print 'y_trainset_cur length', len(y_trainset_cur)
#######################################################

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name+".ckpt")
    print("\nModel restored.")

    for epoch in range(epochs):
        #################################################################
        x_trainset_cur_shuf = []
        x_trainset_pre_shuf = []
        x_trainset_y_pre_shuf = []
        y_trainset_cur_shuf = []

        index_shuf = range(len(x_trainset_cur))
        shuffle(index_shuf)
        for i in index_shuf:
            x_trainset_cur_shuf.append(x_trainset_cur[i])
            x_trainset_pre_shuf.append(x_trainset_pre[i])
            x_trainset_y_pre_shuf.append(x_trainset_y_pre[i])
            y_trainset_cur_shuf.append(y_trainset_cur[i])

        #################################################################
        # #Add some noise to reduse overfitting
        # shift_x = random.uniform(-0.1, 0.1)
        # shift_y = random.uniform(-0.1, 0.1)
        # scale_w = random.uniform(-0.05, 0.05)
        # scale_h = random.uniform(-0.05, 0.05)
        #
        # for i in range(len(x_trainset_cur)):
        #     # print i
        #     y_trainset_cur_shuf_noise[i] = [0,0,0,0]
        #     x_trainset_y_pre_shuf_noise[i] = [0,0,0,0]
        #
        #     # print '\nprevious y_trainset_cur_shuf_noise[i]', y_trainset_cur_shuf_noise[i]
        #     # print 'previous x_trainset_y_pre_shuf_noise[i]', x_trainset_y_pre_shuf_noise[i]
        #
        #     y_trainset_cur_shuf_noise[i][0] = y_trainset_cur_shuf[i][0] + shift_x
        #     y_trainset_cur_shuf_noise[i][1] = y_trainset_cur_shuf[i][1] + shift_y
        #     y_trainset_cur_shuf_noise[i][2] = y_trainset_cur_shuf[i][2] + scale_w
        #     y_trainset_cur_shuf_noise[i][3] = y_trainset_cur_shuf[i][3] + scale_h
        #
        #     x_trainset_y_pre_shuf_noise[i][0] = x_trainset_y_pre_shuf[i][0] + shift_x
        #     x_trainset_y_pre_shuf_noise[i][1] = x_trainset_y_pre_shuf[i][1] + shift_y
        #     x_trainset_y_pre_shuf_noise[i][2] = x_trainset_y_pre_shuf[i][2] + scale_w
        #     x_trainset_y_pre_shuf_noise[i][3] = x_trainset_y_pre_shuf[i][3] + scale_h
        #
        #     # print 'y_trainset_cur_shuf_noise[i]', y_trainset_cur_shuf_noise[i]
        #     # print 'x_trainset_y_pre_shuf_noise[i]\n', x_trainset_y_pre_shuf_noise[i]
        #
        # print 'shift_x', shift_x,' shift_y', shift_y,' scale_w', scale_w,' scale_h', scale_h
        #################################################################

        for batch in range(len(x_trainset_cur)//batch_size):
            batch_x_cur = x_trainset_cur_shuf[batch_size*batch:batch_size*(batch+1)]
            batch_x_pre = x_trainset_pre_shuf[batch_size*batch:batch_size*(batch+1)]
            batch_x_y_pre = x_trainset_y_pre_shuf[batch_size*batch:batch_size*(batch+1)]
            batch_y_cur = y_trainset_cur_shuf[batch_size*batch:batch_size*(batch+1)]
            # print 'length of batch_x: ',len(batch_x)

            logits_out = sess.run(logits, feed_dict={
                x_cur: batch_x_cur,
                x_pre: batch_x_pre,
                x_y_pre: batch_x_y_pre,
                y: batch_y_cur,
                keep_prob: dropout})

            sess.run(optimizer, feed_dict={
                x_cur: batch_x_cur,
                x_pre: batch_x_pre,
                x_y_pre: batch_x_y_pre,
                y: batch_y_cur,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x_cur: batch_x_cur,
                x_pre: batch_x_pre,
                x_y_pre: batch_x_y_pre,
                y: batch_y_cur,
                keep_prob: 1.})

            # Calculate batch loss and accuracy
            loss_L1 = sess.run(cost_L1, feed_dict={
                x_cur: batch_x_cur,
                x_pre: batch_x_pre,
                x_y_pre: batch_x_y_pre,
                y: batch_y_cur,
                keep_prob: 1.})

            valid_acc = sess.run(accuracy, feed_dict={
                x_cur: x_trainset_cur[:test_valid_size],
                x_pre: x_trainset_pre[:test_valid_size],
                x_y_pre: x_trainset_y_pre[:test_valid_size],
                y: y_trainset_cur[:test_valid_size],
                keep_prob: 1.})

            print('\nEpoch {:>2}, Batch {:>3} -'
                  'loss_L1: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss_L1,
                valid_acc))

            print 'logits_out[0]',logits_out[0]
            print "batch_y[0]", batch_y_cur[0]
            print 'learning_rate:', learning_rate , ' dropout: ', dropout


    # Save the variables to disk.
    time = strftime("%Y-%m-%d%H%M", gmtime())
    save_path = saver.save(sess, "./tmp/"+model_name+"re14.ckpt")
    print("Model saved in file: %s" % save_path)

    # # Calculate Test Accuracy
    # test_acc = sess.run(accuracy, feed_dict={
    #     x: mnist.test.images[:test_valid_size],
    #     y: mnist.test.labels[:test_valid_size],
    #     keep_prob: 1.})
    # print('Testing Accuracy: {}'.format(test_acc))
