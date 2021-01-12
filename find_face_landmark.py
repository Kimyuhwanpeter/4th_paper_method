# -*- coding:utf-8 -*-
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import easydict
import os
# AFAD에 대해서도 진행해야된다! 기억해!! --> 현재 진행중 --> 후에는 학습코드를 fasial landmark에 맞게 짜면된다.
FLAGS = easydict.EasyDict({"shape_predict": "C:/Users/Yuhwan/Downloads/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat",
                           
                           "img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD",
                           
                           "nose_text": "C:/Users/Yuhwan/Downloads/AFAD_nose.txt",
                           
                           "eyes_text": "C:/Users/Yuhwan/Downloads/AFAD_eyes.txt",
                           
                           "mouth_text": "C:/Users/Yuhwan/Downloads/AFAD_mouth.txt"})

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.shape_predict)

    data_list = os.listdir(FLAGS.img_path)
    data_list = [FLAGS.img_path + "/" + img for img in data_list]
    
    total_nose_x = []
    total_eyes_x = []
    total_mouth_x = []

    total_nose_y = []
    total_eyes_y = []
    total_mouth_y = []
    count = 0
    for data in data_list:
        
        image = cv2.imread(data)
        image = imutils.resize(image, width=256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        nose_buf_x = []
        eyes_buf_x = []
        mouth_buf_x = []

        nose_buf_y = []
        eyes_buf_y = []
        mouth_buf_y = []

        for (i, rect) in enumerate(rects):
	        # determine the facial landmarks for the face region, then
	        # convert the facial landmark (x, y)-coordinates to a NumPy
	        # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
	        # convert dlib's rectangle to a OpenCV-style bounding box
	        # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	        # loop over the (x, y)-coordinates for the facial landmarks
	        # and draw them on the image
            if len(shape) > 68:
                print("!!!!")

            j = 0
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), 5)
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0

                j += 1
                if j >= 28 and j <= 36: # ?? 9??
                    nose_buf_x.append(x)
                    nose_buf_y.append(y)
                if j >= 37 and j <= 48: # ?? 12??
                    eyes_buf_x.append(x)
                    eyes_buf_y.append(y)
                if j >= 49 and j <= 68: # ?? 20 ?? 
                    mouth_buf_x.append(x)
                    mouth_buf_y.append(y)

            total_nose_x.append(nose_buf_x)
            total_eyes_x.append(eyes_buf_x)
            total_mouth_x.append(mouth_buf_x)

            total_nose_y.append(nose_buf_y)
            total_eyes_y.append(eyes_buf_y)
            total_mouth_y.append(mouth_buf_y)

            break

        # show the output image with the face detections + facial landmarks
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)
        count += 1
        if count % 1000 == 0:
            print(count)
            print("=============================================")
            print("*                  Nose                     *")  
            #print( np.array(np.ceil(np.mean(total_nose_x, 0)), dtype=np.int32) )
            print( np.array(np.mean(total_nose_x, 0)) )
            print( np.array(np.ceil(np.mean(total_nose_y, 0)), dtype=np.int32) )
            print("=============================================")
            print("\n")
            print("=============================================")
            print("*                  Eyes                     *")
            print( np.array(np.ceil(np.mean(total_eyes_x, 0)), dtype=np.int32) )
            print( np.array(np.ceil(np.mean(total_eyes_y, 0)), dtype=np.int32) )
            print("=============================================")
            print("\n")
            print("=============================================")
            print("*                  Mouth                    *")
            print( np.array(np.ceil(np.mean(total_mouth_x, 0)), dtype=np.int32) )
            print( np.array(np.ceil(np.mean(total_mouth_y, 0)), dtype=np.int32) )
            print("=============================================")

    total_nose_x = np.array(np.ceil(np.mean(total_nose_x, 0)), dtype=np.int32)
    total_eyes_x = np.array(np.ceil(np.mean(total_eyes_x, 0)), dtype=np.int32)
    total_mouth_x = np.array(np.ceil(np.mean(total_mouth_x, 0)), dtype=np.int32)

    total_nose_y = np.array(np.ceil(np.mean(total_nose_y, 0)), dtype=np.int32)
    total_eyes_y = np.array(np.ceil(np.mean(total_eyes_y, 0)), dtype=np.int32)
    total_mouth_y = np.array(np.ceil(np.mean(total_mouth_y, 0)), dtype=np.int32)

    write_nose = open(FLAGS.nose_text, "w")
    write_eyes = open(FLAGS.eyes_text, "w")
    write_mouth = open(FLAGS.mouth_text, "w")
    # 이제 된다. 이 부분이 되는지 안되는지 확인만 하면된다.
    for n in range(len(total_nose_x)):
        write_nose.write(str(total_nose_x[n]))
        write_nose.write(" ")
        write_nose.write(str(total_nose_y[n]))
        write_nose.write("\n")
        write_nose.flush()

    for e in range(len(total_eyes_x)):
        write_eyes.write(str(total_eyes_x[e]))
        write_eyes.write(" ")
        write_eyes.write(str(total_eyes_y[e]))
        write_eyes.write("\n")
        write_eyes.flush()

    for m in range(len(total_mouth_x)):
        write_mouth.write(str(total_mouth_x[m]))
        write_mouth.write(" ")
        write_mouth.write(str(total_mouth_y[m]))
        write_mouth.write("\n")
        write_mouth.flush()

if __name__ == "__main__":
    main()