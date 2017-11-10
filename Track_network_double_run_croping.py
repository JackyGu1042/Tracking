import numpy as np
import tensorflow as tf
import cv2
from time import gmtime, strftime
import skvideo.io

model_name = 'model64_4l_2017-11-080843_re15.ckpt'

# Parameters
batch_size = 128

fisrt_file_index =11
end_file_index = fisrt_file_index+1

fisrt_file_index_vot =11
end_file_index_vot = fisrt_file_index_vot#+1
frame_step_vot = 2

initail_frame = 1
end_frame = 600
# Number of samples to calculate validation and accuracy

# Network Parameters
n_output = 4  # Cx,Cy,W,H
dropout = 0.5  # Dropout, probability to keep units

#testing resize image size
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
###############################################################################

    # Layer 1 - 28*28*1 to 14*14*32
    conv_p1 = conv2d(x_pre, weights['wc_p1'], biases['bc_p1'])
    conv_p1 = maxpool2d(conv_p1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv_p2 = conv2d(conv_p1, weights['wc_p2'], biases['bc_p2'])
    conv_p2 = maxpool2d(conv_p2, k=2)

    # Fully connected layer - 7*7*64 to 1024*4
    fc_p1 = tf.reshape(conv_p2, [-1, weights['wd_p1'].get_shape().as_list()[0]])

###############################################################################
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

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
#     .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#Create a saver object which will save all the variables
saver = tf.train.Saver()
#######################################################################

#add one more dimension
image_testset_orginal_cur = []
image_testset_orginal_pre = []

image_testset_cropwidth_pre = []
image_testset_cropheight_pre = []
image_testset_cropxstart_pre = []
image_testset_cropystart_pre = []

x_testset_cur = [] # init numpy array
x_testset_pre = [] #
x_testset_y_pre = []
y_testset_cur = [] # init numpy array

for file_num in range(fisrt_file_index, end_file_index):
    # file_num = 2
    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_test/test_video000"+format(file_num, '02d')+"/test_video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len

    end_frame = frame_len-1 - 6

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
        # image_path_cur = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_cur = cv2.imread(image_path_cur)
        image_testset_orginal_cur.append(image_read_cur)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_test/test_video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)
        image_testset_orginal_pre.append(image_read_pre)

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

        image_testset_cropwidth_pre.append(width)
        image_testset_cropheight_pre.append(height)
        image_testset_cropxstart_pre.append(x_crop_start)
        image_testset_cropystart_pre.append(y_crop_start)

        k1crop_img_cur = image_read_cur[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        k1crop_img_pre = image_read_pre[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

        height_crop, width_crop = k1crop_img_cur.shape[:2]
        print 'width_crop', width_crop, ' height_crop', height_crop

        image_read_resize_cur = cv2.resize(k1crop_img_cur, (width_resize, height_resize))
        image_test_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_test_pre = image_read_resize_pre/255.0-.5

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

        y_test_cur = [x_bl_box_cur, y_bl_box_cur, x_tr_box_cur, y_tr_box_cur]

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

        y_test_pre = [x_bl_box_pre, y_bl_box_pre, x_tr_box_pre, y_tr_box_pre]

        x_testset_cur.append(image_test_cur)
        x_testset_pre.append(image_test_pre)
        x_testset_y_pre.append(y_test_pre)
        y_testset_cur.append(y_test_cur)

#For vot video dataset load
#######################################################################
for file_num in range(fisrt_file_index_vot, end_file_index_vot):

    # text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    text_file = open("./Track_human_dataset/vot_video000"+format(file_num, '02d')+"/groundtruth.txt", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    # text_file = open("./Track_position_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/

    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    end_frame = frame_len/frame_step_vot-1-6
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
        image_testset_orginal_cur.append(image_read_cur)

        # image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_path_pre = './Track_human_dataset/vot_video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        # image_path_pre = './Track_position_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)
        image_testset_orginal_pre.append(image_read_pre)

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

        image_testset_cropwidth_pre.append(width)
        image_testset_cropheight_pre.append(height)
        image_testset_cropxstart_pre.append(x_crop_start)
        image_testset_cropystart_pre.append(y_crop_start)

        k1crop_img_cur = image_read_cur[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        k1crop_img_pre = image_read_pre[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

        height_crop, width_crop = k1crop_img_cur.shape[:2]
        print 'width_crop', width_crop, ' height_crop', height_crop

        image_read_resize_cur = cv2.resize(k1crop_img_cur, (width_resize, height_resize))
        image_test_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_test_pre = image_read_resize_pre/255.0-.5

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

        y_test_cur = [x_bl_box_cur, y_bl_box_cur, x_tr_box_cur, y_tr_box_cur]

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

        y_test_pre = [x_bl_box_pre, y_bl_box_pre, x_tr_box_pre, y_tr_box_pre]

        x_testset_cur.append(image_test_cur)
        x_testset_pre.append(image_test_pre)
        x_testset_y_pre.append(y_test_pre)
        y_testset_cur.append(y_test_cur)

        # image_test_cur = (image_train_cur+0.5)*255.0
        # image_train_pre = (image_train_pre+0.5)*255.0
        # vis = np.concatenate((image_train_cur, image_train_pre), axis=1)
        # cv2.imwrite('./image_result/image_vot_cur_pre_'+format(file_num, '03d')+'_'+format(num_img, '03d')+'.jpg', vis)

print 'x_testset_cur length', len(x_testset_cur)
print 'x_testset_pre length', len(x_testset_pre)
print 'y_testset_cur length', len(y_testset_cur)
#######################################################################

network_out = []
network_out_width_crop = []
network_out_height_crop = []
network_out_xstart_crop = []
network_out_ystart_crop = []
vide_output = []

print 'initail_frame: ', initail_frame
print 'end_frame: ', end_frame

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name)
    print("\nModel restored.")

    cheating = 0
    error_sum = 0
    for num_img in range(initail_frame,end_frame):
        # for num_img in range(test_size):
        if num_img == initail_frame:
            image_input_cur = image_testset_orginal_cur[num_img]
            image_input_pre = image_testset_orginal_pre[num_img]

            y_cur = y_testset_cur[num_img]
            y_pre = x_testset_y_pre[num_img]

            y_crop = y_pre

            x_bl_resize =(y_crop[0]+0.5)*width_resize
            y_bl_resize =(y_crop[1]+0.5)*height_resize
            x_tr_resize = (y_crop[2]+0.5)*width_resize
            y_tr_resize = (y_crop[3]+0.5)*height_resize

            width_crop = image_testset_cropwidth_pre[num_img]
            height_crop = image_testset_cropheight_pre[num_img]
            xstart_crop = image_testset_cropxstart_pre[num_img]
            ystart_crop = image_testset_cropystart_pre[num_img]

            x_bl = x_bl_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop
            y_bl = y_bl_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop
            x_tr = x_tr_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop
            y_tr = y_tr_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop

            x_center = int((x_tr+x_bl)/2)
            y_center = int((y_tr+y_bl)/2)
            width = int(x_tr-x_bl)
            height = int(y_tr-y_bl)

            print 'y_testset_cur:', y_testset_cur[num_img]
        else:
            image_input_cur = image_testset_orginal_cur[num_img]

            y_crop = y_pre
            error_y = y_pre-x_testset_y_pre[num_img]
            print '\nnum:', num_img
            print 'error:',error_y

            x_bl_resize =(y_crop[0]+0.5)*width_resize
            y_bl_resize =(y_crop[1]+0.5)*height_resize
            x_tr_resize = (y_crop[2]+0.5)*width_resize
            y_tr_resize = (y_crop[3]+0.5)*height_resize

            # xstart_crop = image_testset_cropxstart_pre[num_img]
            # ystart_crop = image_testset_cropystart_pre[num_img]

            x_bl = x_bl_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop
            y_bl = y_bl_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop
            x_tr = x_tr_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop
            y_tr = y_tr_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop

            x_center = int((x_tr+x_bl)/2)
            y_center = int((y_tr+y_bl)/2)
            width = int(x_tr-x_bl)
            height = int(y_tr-y_bl)
        # x_input = x_read_cur[initail_frame]

        print 'y_crop',y_crop
        print 'x_center', x_center,' y_center', y_center, ' width', width, ' height', height
        print 'y_center - int(height*k1/2)', y_center - int(height*k1/2),' y_center + int(height*k1/2)', y_center + int(height*k1/2)
        print ' x_center - int(width*k1/2)', x_center - int(width*k1/2), ' x_center + int(width*k1/2)', x_center + int(width*k1/2)


        y_crop_s = max(0,y_center - int(height*k1/2))
        y_crop_e = y_center + int(height*k1/2)
        x_crop_s = max(0,x_center - int(width*k1/2))
        x_crop_e = x_center + int(width*k1/2)

        print 'y_crop_s', y_crop_s,' y_crop_e', y_crop_e, ' x_crop_s', x_crop_s, ' x_crop_e', x_crop_e

        x_input_k1crop_cur = image_input_cur[y_crop_s:y_crop_e, x_crop_s:x_crop_e]
        x_input_k1crop_resize_cur = cv2.resize(x_input_k1crop_cur, (width_resize, height_resize))
        x_input_k1crop_resize_cur = x_input_k1crop_resize_cur/255.0-.5

        x_input_k1crop_pre = image_input_pre[y_crop_s:y_crop_e, x_crop_s:x_crop_e]
        x_input_k1crop_resize_pre = cv2.resize(x_input_k1crop_pre, (width_resize, height_resize))
        x_input_k1crop_resize_pre = x_input_k1crop_resize_pre/255.0-.5

        # cv2.imshow('image_origin_cur'+format(num_img, '03d'),image_input_cur)
        # cv2.imshow('image_croped_cur'+format(num_img, '03d'),x_input_k1crop_resize_cur)
        # cv2.imshow('image_croped_pre'+format(num_img, '03d'),x_input_k1crop_resize_pre)

        logits_out = sess.run(logits, feed_dict={
            x_cur: [x_input_k1crop_resize_cur],
            x_pre: [x_input_k1crop_resize_pre],
            x_y_pre: [y_pre],
            y: [[1,1,1,1]],
            keep_prob: 1})

        y_cur = logits_out[0]


        width_height_tunning = 1
        x_y_startcrop_tunning = 30
        width_rescale = (y_cur[2]-y_cur[0])/(y_pre[2]-y_pre[0])
        height_rescale = (y_cur[3]-y_cur[1])/(y_pre[3]-y_pre[1])
        width_crop = width_crop*width_rescale*width_height_tunning
        height_crop = height_crop*height_rescale*width_height_tunning
        xstart_crop = xstart_crop + (y_cur[0]+y_cur[2]-0)*0.5*(float(width_crop*k1)/float(width_resize))*x_y_startcrop_tunning
        ystart_crop = ystart_crop + (y_cur[1]+y_cur[3]-0)*0.5*(float(height_crop*k1)/float(height_resize))*x_y_startcrop_tunning
        print '(y_cur[0]-y_pre[0])*(float(width_crop*k1)/float(width_resize))' , (y_cur[0]-y_pre[0])*(float(width_crop*k1)/float(width_resize))
        print '(y_cur[1]-y_pre[1])*(float(height_crop*k1)/float(height_resize))', (y_cur[1]-y_pre[1])*(float(height_crop*k1)/float(height_resize))
        img_rec_out_cur = cv2.rectangle(x_input_k1crop_resize_cur,(int((y_cur[0]+0.5)*width_resize),int((y_cur[1]+0.5)*height_resize)),(int((y_cur[2]+0.5)*width_resize),int((y_cur[3]+0.5)*height_resize)),(0,255,0),1)
        img_rec_out_pre = cv2.rectangle(x_input_k1crop_resize_pre,(int((y_pre[0]+0.5)*width_resize),int((y_pre[1]+0.5)*height_resize)),(int((y_pre[2]+0.5)*width_resize),int((y_pre[3]+0.5)*height_resize)),(0,255,0),1)

        # cv2.putText(img_rec_out_cur,"Current frame", (int((y_cur[0]+0.5)*width_resize),int((y_cur[1]+0.5)*width_resize)), cv2.FONT_HERSHEY_SIMPLEX, 0.1, 255)
        # cv2.putText(img_rec_out_pre,"previous frame", (int((y_pre[0]+0.5)*width_resize),int((y_pre[1]+0.5)*width_resize)), cv2.FONT_HERSHEY_SIMPLEX, 0.05, 255)

        x_input_k1crop_resize_cur = (img_rec_out_cur+0.5)*255.0
        x_input_k1crop_resize_pre = (img_rec_out_pre+0.5)*255.0
        vis = np.concatenate((x_input_k1crop_resize_cur, x_input_k1crop_resize_pre), axis=1)
        cv2.imwrite('./image_result/image_croping_pre_cur'+format(num_img, '03d')+'.jpg', vis)

        y_pre = y_cur
        network_out.append(logits_out[0])
        network_out_width_crop.append(width_crop)
        network_out_height_crop.append(height_crop)
        network_out_xstart_crop.append(xstart_crop)
        network_out_ystart_crop.append(ystart_crop)

        image_input_pre = image_input_cur
        print 'logits_out'+format(num_img, '03d'),logits_out

for num_img in range(initail_frame,end_frame):

    image_test = image_testset_orginal_cur[num_img]
    y_out = network_out[num_img-initail_frame]
    # print num_img,':',y_out

    x_bl_resize =(y_out[0]+0.5)*width_resize
    y_bl_resize =(y_out[1]+0.5)*height_resize
    x_tr_resize = (y_out[2]+0.5)*width_resize
    y_tr_resize = (y_out[3]+0.5)*height_resize

    width_crop = network_out_width_crop[num_img-initail_frame]
    height_crop = network_out_height_crop[num_img-initail_frame]
    xstart_crop = network_out_xstart_crop[num_img-initail_frame]
    ystart_crop = network_out_ystart_crop[num_img-initail_frame]

    x_bl = int(x_bl_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop)
    y_bl = int(y_bl_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop)
    x_tr = int(x_tr_resize*(float(width_crop*k1)/float(width_resize)) + xstart_crop)
    y_tr = int(y_tr_resize*(float(height_crop*k1)/float(height_resize)) + ystart_crop)

    img_rec_out = cv2.rectangle(image_test,(x_bl,y_bl),(x_tr,y_tr),(0,255,0),2)
    print ' x_bl',x_bl,' y_bl',y_bl,' x_tr',x_tr, ' y_tr',y_tr
    print ' xstart_crop',xstart_crop,' ystart_crop',ystart_crop
    # x_lt_crop = int(y_out[0]-y_out[2]*k1/2)
    # y_lt_crop = int(y_out[1]-y_out[3]*k1/2)
    # x_rb_crop = int(y_out[0]+y_out[2]*k1/2)
    # y_rb_crop = int(y_out[1]+y_out[3]*k1/2)
    # img_rec_out = cv2.rectangle(image_test,(x_lt_crop,y_lt_crop),(x_rb_crop,y_rb_crop),(0,200,0),1)

    outputdata = img_rec_out.astype(np.uint8)
    vide_output.append(outputdata)
    # cv2.imshow('image_croping'+format(num_img, '03d'),img_rec_out)

skvideo.io.vwrite("./video_result/track_video_croping_d4l_reback"+format(fisrt_file_index, '02d')+".mp4", vide_output)
print "video built"

print "\nmodel_name",model_name
# print "cheating time",cheating
# print 'error_sum',error_sum, ' sum of error', sum(i*i for i in error_sum)

cv2.waitKey(0)
