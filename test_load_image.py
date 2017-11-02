import cv2
import numpy as np
from time import gmtime, strftime
import skvideo.io

width_resize = 64
height_resize = 64

x_trainset_cur = [] # init numpy array
x_trainset_pre = [] #
y_trainset_cur = [] # init numpy array

#######################################################################
for file_num in range(1,42):
    # file_num = 2
    text_file = open("./Track_dataset/video000"+format(file_num, '02d')+"/video000"+format(file_num, '02d')+".ann", "r")#./Home/Desktop/Tracking/Dataset/vot2014/drunk/
    groundtruth_frame = text_file.readlines()
    text_file.close()

    frame_len = len(groundtruth_frame)
    print 'len(groundtruth_frame):', frame_len
    # print groundtruth_frame
    k1 = 2
    vide_output_cur = []
    vide_output_pre = []

    for num_img in range(1, frame_len):
        y_temp_cur = groundtruth_frame[num_img]
        y_temp_cur = y_temp_cur.split()
        y_temp_cur_float = [float(i) for i in y_temp_cur]

        y_temp_pre = groundtruth_frame[num_img-1]
        y_temp_pre = y_temp_pre.split()
        y_temp_pre_float = [float(i) for i in y_temp_pre]

        # y_temp = groundtruth_frame[num_img]
        # y_temp = y_temp.split()
        # y_temp_float = [float(i) for i in y_temp]
        frame_index_cur = int(y_temp_cur_float[0])
        frame_index_pre = int(y_temp_pre_float[0])

        print 'file_num', file_num
        print 'frame_index_cur', frame_index_cur , ' frame_index_pre', frame_index_pre

        image_path_cur = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_cur, '03d') + '.jpg'
        image_read_cur = cv2.imread(image_path_cur)

        image_path_pre = './Track_dataset/video000'+format(file_num, '02d')+'/00000' + format(frame_index_pre, '03d') + '.jpg'
        image_read_pre = cv2.imread(image_path_pre)

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

        # if frame_index <= 4 and frame_index >= 4:
        #     img_cir = cv2.circle(image_read,(int(y_temp_float[1]),int(y_temp_float[2])), 5, (0,0,255), -1)
        #     img_cir = cv2.circle(image_read,(int(y_temp_float[3]),int(y_temp_float[4])), 5, (0,0,200), -1)
        #     img_cir = cv2.circle(image_read,(int(y_temp_float[5]),int(y_temp_float[6])), 5, (0,0,155), -1)
        #     img_cir = cv2.circle(image_read,(int(y_temp_float[7]),int(y_temp_float[8])), 5, (0,0,55), -1)
        #     cv2.imshow("cir image_"+format(frame_index, '02d'),img_cir)
        #     if frame_index == 4:
        #         cv2.waitKey(0)

        image_read_resize_cur = cv2.resize(k1crop_img_cur, (width_resize, height_resize))
        image_train_cur = image_read_resize_cur/255.0-.5

        image_read_resize_pre = cv2.resize(k1crop_img_pre, (width_resize, height_resize))
        image_train_pre = image_read_resize_pre/255.0-.5

        #use current y as turth value
        height_f = abs((y_temp_cur_float[6]-y_temp_cur_float[4])/width_original-0.5) #xc
        width_f = abs((y_temp_cur_float[1]-y_temp_cur_float[3])/height_original-0.5) #yc
        x_center_f = ((y_temp_cur_float[1]+y_temp_cur_float[3])/2.0)/width_original-0.5 #width
        y_center_f = ((y_temp_cur_float[2]+y_temp_cur_float[8])/2.0)/height_original-0.5 #height

        y_train_cur = [x_center_f, y_center_f, width_f, height_f]

        # y_train_4[0] = y_train_4[0]/width_original-0.5 #xc
        # y_train_4[1] = y_train_4[1]/height_original-0.5 #yc
        # y_train_4[2] = y_train_4[2]/width_original-0.5 #width
        # y_train_4[3] = y_train_4[3]/height_original-0.5 #height

        # x_train.append(image_train)
        # y_train.append(y_train_4)

        # x_lt = int(y_train_cur[0]-y_train_cur[2]/2)
        # y_lt = int(y_train_cur[1]-y_train_cur[3]/2)
        # x_rb = int(y_train_cur[0]+y_train_cur[2]/2)
        # y_rb = int(y_train_cur[1]+y_train_cur[3]/2)
        # img_rec_out = cv2.rectangle(image_read,(x_lt,y_lt),(x_rb,y_rb),(0,255,0),2)

        # outputdata = img_rec_out.astype(np.uint8)
        # outputdata_cur = image_read_resize_cur.astype(np.uint8)
        # vide_output_cur.append(outputdata_cur)
        #
        # outputdata_pre = image_read_resize_pre.astype(np.uint8)
        # vide_output_pre.append(outputdata_pre)

        x_trainset_cur.append(image_train_cur)
        x_trainset_pre.append(image_train_pre)
        y_trainset_cur.append(y_train_cur)

    # skvideo.io.vwrite("Dataset_video_cur_"+format(file_num, '02d')+".mp4", vide_output_cur)
    # skvideo.io.vwrite("Dataset_video_pre_"+format(file_num, '02d')+".mp4", vide_output_pre)
    # print "video built"

print 'x_trainset_cur length', len(x_trainset_cur)
print 'x_trainset_pre length', len(x_trainset_pre)
print 'y_trainset_cur length', len(y_trainset_cur)
# cv2.waitKey(0)
# cv2.waitKey(0)




# k1=2
# y = [200.35, 159.32, 200.35, 113.74, 245.48, 113.74, 245.48, 159.32]
#
# height = int(y[1]-y[3])
# width = int(y[6]-y[0])
# x_coner = int(y[2])
# y_coner = int(y[3])
# x_center = int((y[6]+y[0])/2.0)
# y_center = int((y[1]+y[3])/2.0)
# y_4 = [x_center, y_center, width, height]
#
# # border widths; I set them all to 150
# border_wt = int((100 - width*k1)/2)
# border_wb = 100 - border_wt - width*k1
# border_hl = int((100 - height*k1)/2)
# border_hr = 100 - border_hl - height*k1
# top, bottom, left, right = [border_wt,border_wb,border_hl,border_hr]
# #######################################################################
#
# print 'w:',width,' h:',height,' x_n:',x_coner,' y_n:',y_coner,' x_c:', x_center,' y_c:',y_center
# crop_img = image_test[y_center - int(height/2):y_center + int(height/2), x_center - int(width/2):x_center + int(width/2)]
# cv2.imshow("crop image",crop_img)
#
# k1crop_img = image_test[y_center - int(height*k1/2):y_center + int(height*k1/2), x_center - int(width*k1/2):x_center + int(width*k1/2)]
# cv2.imshow("*k1 crop image",k1crop_img)
#
# k1crop_img_with_border = cv2.copyMakeBorder(k1crop_img, top, bottom, left, right, cv2.BORDER_REFLECT)
# cv2.imshow("*k1 crop image with border",k1crop_img_with_border)
#
# # height = np.size(k1crop_img_with_border, 0)
# # width = np.size(k1crop_img_with_border, 1)
# # channel = np.size(k1crop_img_with_border, 2)
# # print 'k1crop_img_with_border: ',width,"x",height,"x",channel
# #######################################################################
#
# # copy image to display all 4 variations
# horizontal_img = image_test.copy()
# vertical_img = image_test.copy()
# both_img = image_test.copy()
#
# # flip img horizontally, vertically,
# # and both axes with flip()
# horizontal_img = cv2.flip( image_test, 0 )
# vertical_img = cv2.flip( image_test, 1 )
# both_img = cv2.flip( image_test, -1 )
#
# cv2.imshow( "Original image", image_test )
#
# y_4 = [x_center, y_center, width, height]
# o_img = image_test[y_4[1] - int(y_4[3]*k1/2):y_4[1] + int(y_4[3]*k1/2), y_4[0] - int(y_4[2]*k1/2):y_4[0] + int(y_4[2]*k1/2)]
# # o_img = image_test[y_center - int(height*k1/2):y_center + int(height*k1/2), x_center - int(width*k1/2):x_center + int(width*k1/2)]
# print 'x_c:', x_center,' y_c:',y_center, ' w:',width,' h:',height
# cv2.imshow("Original",o_img)
#
# cv2.imshow( "Horizontal flip", horizontal_img )
# y_4_h = [x_center, y_center, width, height]
# y_4_h[1] = 240 - y_4[1]
# h_img = horizontal_img[y_4_h[1] - int(y_4_h[3]*k1/2):y_4_h[1] + int(y_4_h[3]*k1/2), y_4_h[0] - int(y_4_h[2]*k1/2):y_4_h[0] + int(y_4_h[2]*k1/2)]
# cv2.imshow("Horizontal",h_img)
#
# cv2.imshow( "Vertical flip", vertical_img )
# y_4_v = [x_center, y_center, width, height]
# y_4_v[0] = 320 - y_4[0]
# v_img = vertical_img[y_4_v[1] - int(y_4_v[3]*k1/2):y_4_v[1] + int(y_4_v[3]*k1/2), y_4_v[0] - int(y_4_v[2]*k1/2):y_4_v[0] + int(y_4_v[2]*k1/2)]
# print 'y_4:',y_4
# print 'y_4_v:',y_4_v
# cv2.imshow("Vertical",v_img)
#
# cv2.imshow( "Both flip", both_img )
# y_4_b = [x_center, y_center, width, height]
# y_4_b[0] = 320 - y_4[0]
# y_4_b[1] = 240 - y_4[1]
# b_img = both_img[y_4_b[1] - int(y_4_b[3]*k1/2):y_4_b[1] + int(y_4_b[3]*k1/2), y_4_b[0] - int(y_4_b[2]*k1/2):y_4_b[0] + int(y_4_b[2]*k1/2)]
# cv2.imshow("Both",b_img)
#
# #######################################################################
# img_rec = cv2.rectangle(image_test,(int(y[2]),int(y[3])),(int(y[6]),int(y[7])),(0,255,0),2)
# # cv2.imshow("rec image",img_rec)
#
# img_cir = cv2.circle(image_test,(int(y[0]),int(y[1])), 5, (0,0,255), -1)
# img_cir = cv2.circle(image_test,(int(y[2]),int(y[3])), 5, (0,0,200), -1)
# img_cir = cv2.circle(image_test,(int(y[4]),int(y[5])), 5, (0,0,155), -1)
# img_cir = cv2.circle(image_test,(int(y[6]),int(y[7])), 5, (0,0,55), -1)
# cv2.imshow("cir image",img_cir)
#
# img_rec_v = cv2.rectangle(vertical_img,(y_4_v[0]-y_4_v[2]/2,y_4_v[1]-y_4_v[3]/2),(y_4_v[0]+y_4_v[2]/2,y_4_v[1]+y_4_v[3]/2),(0,255,0),2)
# cv2.imshow("rec image",img_rec_v)
# #######################################################################
#
# time = strftime("%Y-%m-%d%H%M", gmtime())
# print time
# cv2.waitKey(0)
