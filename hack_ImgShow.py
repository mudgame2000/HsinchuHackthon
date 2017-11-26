import cv2
import os          
import numpy as np
from matplotlib import pyplot as plt

cv2.ocl.setUseOpenCL(False)

##cap = cv2.VideoCapture('2015.10.01 07.33.55.avi')
cap = cv2.VideoCapture('2015.05.08 07.54.30.avi')

track_width,track_height = 280,180
frame_per_seconds =2
frame_cnt =0
frame_write_flag ="N"
display_num=0
display_time=0

CA_param=50 
CA_min_area=500

# resize param
dst_cols_5x3=500
dst_cols_3x5=300

img_output_path='/home/m10516013/git/caffe/prepare/img_out'
img_output_file_cnt=0

history = 80;
#dist2Threshold = 5000;
dist2Threshold = 500;
#detectShadows = False;
detectShadows = True;
pKNN = cv2.createBackgroundSubtractorKNN(history,dist2Threshold,detectShadows)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

## test pKNN = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    if ret == True:
        frame_cnt = frame_cnt + 1
        cv2.imshow('(1). orign image',frame)
        print("-----frame_cnt-----> ",frame_cnt)

        #----------------------------------+
        #  ROI: skip bad feature of image  |
        #----------------------------------+
        img_rows, img_cols, img_channels = frame.shape
        print("frame -> rows,cols,channels", img_rows,img_cols,img_channels)

        ROI = frame[(img_rows/3):(img_rows), :img_cols]
        cv2.imshow('(2). ROI image',ROI)

        #--------------------------+
        #  resize image rows/cols  |
        #--------------------------+
        img_wh_rate=float(img_rows)/float(img_cols)
        print("img_wh_rate", img_wh_rate)

        if ( img_wh_rate <= 1.0 ):
            scaling_rate=float(dst_cols_5x3)/float(img_cols)
            scaling_cols=round(img_cols*scaling_rate)
            scaling_rows=round(img_rows*scaling_rate)
            print("img_wh_rate <= 1.0:scaling -> rows,cols", scaling_rows, scaling_cols)
        else:
            scaling_rate=float(dst_cols_3x5)/float(img_cols)
            scaling_cols=round(img_cols*scaling_rate)
            scaling_rows=round(img_rows*scaling_rate)
            print("scaling -> rows,cols", scaling_rows,scaling_cols)
            print("img_wh_rate > 1.0:scaling -> rows,cols", scaling_rows, scaling_cols)
        #end-if
        ##res = cv2.resize(img_dst,None,fx=rate_x, fy=rate_y, interpolation = cv2.INTER_CUBIC)
        img_res = cv2.resize(ROI,(int(scaling_cols),int(scaling_rows)),0,0, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('(3). resize image',img_res)
        
        #----------------------------+
        # track window rows/cols set |
        #----------------------------+
        dst_rows,dst_cols,dst_channels=img_res.shape
        #track_y=(dst_rows/2)    
        #track_x=(dst_cols/2)-(track_width/2)    

        track_y_min=144   
        track_x_min=10    
        track_y_max=dst_rows - track_y_min
        track_x_max=dst_cols - track_x_min

        ##track_y_min=400   
        ##track_x_min=10    
        ##track_y_max=dst_rows - 10
        ##track_x_max=dst_cols - track_x_min

        #-------------------+
        #  frame substract  |
        #-------------------+
        fgmask = pKNN.apply(img_res)
        cv2.imshow('(4). fgmask image',fgmask)

        #-----------------+
        # opening closing |
        #-----------------+
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        #--------+
        # dilate |
        #--------+
        _,th = cv2.threshold(closing,254,255,cv2.THRESH_BINARY)
        dilated = cv2.dilate(th,kernel,iterations = 2 )

        ##print("--->frame_cnt%frame_per_seconds",(frame_cnt%frame_per_seconds))
        ##if ( frame_cnt >= frame_per_seconds ):
        ##     print("----img_write_flg = Y-------")
        ##     frame_write_flag="Y"
        ##     frame_cnt=0

        #-----------------------------------+
        # skip first 10 frame for bad frame |
        #-----------------------------------+
        if ( frame_cnt > 10 ):

            frame_write_flag="N"

            img_save = img_res
            cv2.imshow('(4.5). save image',img_save)

            #---------------+
            # find contours |
            #---------------+
            img3 , contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow('(5). contours image',img3)

            for c in contours:
                (x,y,w,h) = cv2.boundingRect(c)
                if ( cv2.contourArea(c) > CA_param and w*h > CA_min_area and frame_write_flag == "N" ):
                    if ( x > track_x_min and y > track_y_min and x+w < track_x_max and y+h < track_y_max):
                        print ("---------------- .xml file head -----------------", w , h, w*h)

                        frame_write_flag="Y"
                        img_output_file_cnt= img_output_file_cnt + 1

                        if ( len(str(img_output_file_cnt)) == 1 ):
                            img_out_flnm = '00000'     + str(img_output_file_cnt)
                        elif ( len(str(img_output_file_cnt)) == 2 ):
                           img_out_flnm = '0000' + str(img_output_file_cnt)
                        elif ( len(str(img_output_file_cnt)) == 3 ):
                           img_out_flnm = '000'  + str(img_output_file_cnt)
                        elif ( len(str(img_output_file_cnt)) == 4 ):
                           img_out_flnm = '00'   + str(img_output_file_cnt)
                        elif ( len(str(img_output_file_cnt)) == 5 ):
                           img_out_flnm = '0'    + str(img_output_file_cnt)

                        #-----------------+
                        # write .jpg file |
                        #-----------------+
                        print("----------> img_out_flnm----------> ", img_out_flnm )
                        ##cv2.imwrite(os.path.join(img_output_path,img_out_flnm + ".jpg"),img_save)
    
                        #------------------------------------+
                        # write annotation .xml file  header |
                        #------------------------------------+
                        fo = open(os.path.join(img_output_path,img_out_flnm + ".xml"),"w+")
                        fo.write("<annotation>\n")
                        fo.write("    <folder>JPEGImages</folder>\n")
                        fo.write("    <filename>" + img_out_flnm + ".jpg</filename>\n")
                        fo.write("    <path>/home/m10516013/data/VOCdevkit/VOC2007/JPEGImages/" + img_out_flnm + ".jpg</path>\n")
                        fo.write("    <source>\n")
                        fo.write("        <database>Unknown</database>\n")
                        fo.write("    </source>\n")
                        fo.write("    <size>\n")
                        fo.write("        <width>" + str(dst_cols) + "</width>\n")
                        fo.write("        <height>" + str(dst_rows) + "</height>\n")
                        fo.write("        <depth>" + str(dst_channels) + "</depth>\n")
                        fo.write("    </size>\n")
                        fo.write("    <segmented>0</segmented>\n")
                    #end-if
                #end-if

                if ( cv2.contourArea(c) > CA_param and w*h > CA_min_area ):
                    if ( x > track_x_min and y > track_y_min and x+w < track_x_max and y+h < track_y_max):
                        print ("---------------- .xml file object -----------------", w,h,w*h )
                        #-----------------------------------------+
                        # write annotation .xml file body(object) |
                        #-----------------------------------------+
                        fo.write("    <object>\n")
                        fo.write("        <name>car</name>\n")
                        fo.write("        <pose>Unspecified</pose>\n")
                        fo.write("        <truncated>0</truncated>\n")
                        fo.write("        <difficult>0</difficult>\n")
                        fo.write("        <bndbox>\n")
                        fo.write("            <xmin>" + str(x) + "</xmin>\n")
                        fo.write("            <ymin>" + str(y) + "</ymin>\n")
                        fo.write("            <xmax>" + str(x+w) + "</xmax>\n")
                        fo.write("            <ymax>" + str(y+h) + "</ymax>\n")
                        fo.write("        </bndbox>\n")
                        fo.write("    </object>\n")

                        #----------------+
                        # draw rectangle |
                        #----------------+
                        cv2.rectangle(img_res,(x,y), (x+w,y+h), (255,0,0),1)

                        #---------------+
                        # draw contours |
                        #---------------+
                        cv2.drawContours(img_res,c,-1,(0,255,0),1)

                        #-------------+
                        # draw circle |
                        #-------------+
                        M = cv2.moments(c)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(img_res,(cx,cy), 3, (0,0,255), 1)
                        display_num = display_num + 1

                    #end-if
                #end-if
            #end-for-contour

            if ( frame_write_flag == "Y" ):
                print("--------------- imwrite ---------------")

                #-----------------+
                # write .jpg file |
                #-----------------+
                cv2.imwrite(os.path.join(img_output_path,img_out_flnm + ".jpg"),img_res)

                #----------------------------------+
                # write annotation .xml file trail |
                #----------------------------------+
                fo.write("</annotation>\n")

                #-----------------+
                # close .xml file |
                #-----------------+
                fo.close()

            #end-if

            #####################
            # draw track window #
            #####################
            if ( frame_cnt % 30 == 0 ):
                display_time = display_time + 1
            cv2.putText(img_res,"Time(sec):"+str(display_time)+",COUNTER:"+ str(display_num), (30,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,175),1)
            cv2.rectangle(img_res,(track_x_min,track_y_min), (track_x_max,track_y_max), (0,0,225),2)
            cv2.imshow('(6). detection image',img_res)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            #end-if
        #end-if
    else:
        break
    #end-if

cap.release()
cv2.destroyAllWindows()
