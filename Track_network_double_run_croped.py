import numpy as np
import tensorflow as tf
import cv2
from time import gmtime, strftime
import skvideo.io

model_name = 'model64_4l_2017-10-310907re1.ckpt'

# Parameters
batch_size = 128

fisrt_file_index = 1
end_file_index = 2

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
k1=2

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], mean = mu, stddev = sigma)),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], mean = mu, stddev = sigma)),
    'wd1': tf.Variable(tf.random_normal([(width_resize/4)**2*64, 1024*2], mean = mu, stddev = sigma)),

    'wc_p1': tf.Variable(tf.random_normal([5, 5, 3, 32], mean = mu, stddev = sigma)),
    'wc_p2': tf.Variable(tf.random_normal([5, 5, 32, 64], mean = mu, stddev = sigma)),
    'wd_p1': tf.Variable(tf.random_normal([(width_resize/4)**2*64, 1024*2], mean = mu, stddev = sigma)),

    'wd_y_pre1': tf.Variable(tf.random_normal([4, 1024*2], mean = mu, stddev = sigma)),

    'wd2': tf.Variable(tf.random_normal([1024*2, 1024], mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.random_normal([1024, n_output], mean = mu, stddev = sigma))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32], mean = mu, stddev = sigma)),
    'bc2': tf.Variable(tf.random_normal([64], mean = mu, stddev = sigma)),

    'bc_p1': tf.Variable(tf.random_normal([32], mean = mu, stddev = sigma)),
    'bc_p2': tf.Variable(tf.random_normal([64], mean = mu, stddev = sigma)),

    'bd1': tf.Variable(tf.random_normal([1024*2], mean = mu, stddev = sigma)),
    'bd2': tf.Variable(tf.random_normal([1024], mean = mu, stddev = sigma)),
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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

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

x_testset_cur = [] # init numpy array
x_testset_pre = [] #
x_testset_y_pre = []
y_testset_cur = [] # init numpy array

for file_num in range(fisrt_file_index, end_file_index):
    # file_num = 2
    text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len

    for num_img in range(1, frame_len):
        y_temp_cur = groundtruth_frame[num_img]
        y_temp_cur = y_temp_cur.split()
        y_temp_cur_float = [float(i) for i in y_temp_cur]

        y_temp_pre = groundtruth_frame[num_img-1]
        y_temp_pre = y_temp_pre.split()
        y_temp_pre_float = [float(i) for i in y_temp_pre]

        frame_index_cur = int(y_temp_cur_float[0])
        frame_index_pre = int(y_temp_pre_float[0])

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_cur = cv2.imread(image_path_cur)
        image_testset_orginal_cur.append(image_read_cur)

        image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)
        image_testset_orginal_pre.append(image_read_pre)

        height_original, width_original = image_read_cur.shape[:2]
        print 'width_original', width_original, ' height_original', height_original

        #Use previous frame to crop previous frame and current frame, and use current y as turth value
        height = abs(int(y_temp_pre_float[6]-y_temp_pre_float[4]))
        width = abs(int(y_temp_pre_float[1]-y_temp_pre_float[3]))
        x_center = int((y_temp_pre_float[1]+y_temp_pre_float[3])/2.0)
        y_center = int((y_temp_pre_float[2]+y_temp_pre_float[8])/2.0)

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
        image_test_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_test_pre = image_read_resize_pre/255.0-.5

        #use current y as turth value
        height_f = abs((y_temp_cur_float[6]-y_temp_cur_float[4])/width_original-0.5) #xc
        width_f = abs((y_temp_cur_float[1]-y_temp_cur_float[3])/height_original-0.5) #yc
        x_center_f = ((y_temp_cur_float[1]+y_temp_cur_float[3])/2.0)/width_original-0.5 #width
        y_center_f = ((y_temp_cur_float[2]+y_temp_cur_float[8])/2.0)/height_original-0.5 #height

        y_test_cur = [x_center_f, y_center_f, width_f, height_f]

        #use current y as turth value
        height_fp = abs((y_temp_pre_float[6]-y_temp_pre_float[4])/width_original-0.5) #xc
        width_fp = abs((y_temp_pre_float[1]-y_temp_pre_float[3])/height_original-0.5) #yc
        x_center_fp = ((y_temp_pre_float[1]+y_temp_pre_float[3])/2.0)/width_original-0.5 #width
        y_center_fp = ((y_temp_pre_float[2]+y_temp_pre_float[8])/2.0)/height_original-0.5 #height

        y_test_pre = [x_center_fp, y_center_fp, width_fp, height_fp]

        x_testset_cur.append(image_test_cur)
        x_testset_pre.append(image_test_pre)
        x_testset_y_pre.append(y_test_pre)
        y_testset_cur.append(y_test_cur)


print 'x_testset_cur length', len(x_testset_cur)
print 'x_testset_pre length', len(x_testset_pre)
print 'y_testset_cur length', len(y_testset_cur)
#######################################################################

vide_output = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name)
    print("\nModel restored.")

    batch_x_cur = x_testset_cur[initail_frame:end_frame]
    batch_x_pre = x_testset_pre[initail_frame:end_frame]
    batch_x_y_pre = x_testset_y_pre[initail_frame:end_frame]
    batch_y_cur = y_testset_cur[initail_frame:end_frame]

    logits_out = sess.run(logits, feed_dict={
        x_cur: batch_x_cur,
        x_pre: batch_x_pre,
        x_y_pre: batch_x_y_pre,
        y: batch_y_cur,
        keep_prob: 1})

    print 'logits_out[0]',logits_out[0]
    print "batch_y[0]", batch_y_cur[0]
#######################################################################

for num_img in range(initail_frame,end_frame):
    # num_img = test_size-num_img-1
    image_test = image_testset_orginal_cur[initail_frame]
    y_out = logits_out[num_img-initail_frame]
    # y_out = y_test[num_img-1]
    print y_out

    y_out[0] = (y_out[0]+0.5)*width_original #xc
    y_out[1] = (y_out[1]+0.5)*height_original #yc
    y_out[2] = (y_out[2]+0.5)*width_original #width
    y_out[3] = (y_out[3]+0.5)*height_original #height

    x_lt = int(y_out[0]-y_out[2]/2)
    y_lt = int(y_out[1]-y_out[3]/2)
    x_rb = int(y_out[0]+y_out[2]/2)
    y_rb = int(y_out[1]+y_out[3]/2)
    img_rec_out = cv2.rectangle(image_test,(x_lt,y_lt),(x_rb,y_rb),(0,255,0),2)

    outputdata = img_rec_out.astype(np.uint8)
    vide_output.append(outputdata)
    # cv2.imshow('image_croped'+format(num_img, '03d'),img_rec_out)

skvideo.io.vwrite("drunk_video_croped_d4l.mp4", vide_output)
print "video built"

cv2.waitKey(0)
