'''
-*- coding:utf-8 -*-
@file        armor recognition.py
@author      北京工业大学 PIP 战队 23 杨文远
@version     v3.8
@details

'''
import os
import sys
import cv2
import numpy as np
from typing import Tuple
from ctypes import *
 


def contourDrawing(image: np.ndarray , Color:str) -> Tuple[np.ndarray, list] :
    """Function summary : draw the contours of image
    
    Args:
        image(np.ndarray): raw image.
        
    Returns:
        np.ndarray: image of a contour around the light bar
        list: contours on image.
    
    """
    exposure_factor = 0.621  
    low_exposure_image = cv2.multiply(image, (1-exposure_factor))   #减小曝光度
    if Color == "blue" :
        lower_blue = np.array([30, 30, 80])
        upper_blue = np.array([256, 256, 256])
        hsv_image = cv2.cvtColor(low_exposure_image, cv2.COLOR_BGR2HSV)  #将低曝光度的图片从RGB颜色空间转化到HSV颜色空间
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)    #创建掩膜
        blue_objects = cv2.bitwise_and(image, image, mask=mask)  #将灯条的蓝色筛选出来
        gray = cv2.cvtColor(blue_objects, cv2.COLOR_BGR2GRAY)  #转化成灰度图
    if Color == "red" :
        lower_red = np.array([80, 30, 30])
        upper_red = np.array([256, 256, 256])
        hsv_image = cv2.cvtColor(low_exposure_image, cv2.COLOR_BGR2HSV)  #将低曝光度的图片从RGB颜色空间转化到HSV颜色空间
        mask = cv2.inRange(hsv_image, lower_red, upper_red)    #创建掩膜
        red_objects = cv2.bitwise_and(image, image, mask=mask)  #将灯条的蓝色筛选出来
        gray = cv2.cvtColor(red_objects, cv2.COLOR_BGR2GRAY)  #转化成灰度图
    ret , binary = cv2.threshold(gray,180,255,cv2.THRESH_BINARY) #二值化，进一步清晰化灯条
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(image, contours, -1, (2, 0, 255), 1)#将轮廓勾勒
    return image, contours 
    
def getCenter(contours: list) -> Tuple[list, list] :
    """Function summary : get each contour's center
    
    Args:
        contours(list) : contours on image.
        
    Returns:
        list: center of each contour.
        list: contours after the first screening
    
    """
    i=0
    center = []
    valid_contours = []
    for contour in contours : 
        M = cv2.moments(contour)
        if M["m00"] != 0: #图像的零阶矩不等于0，将白色的轮廓筛除
            i=i+1
            center_x = (M["m10"]/M["m00"])
            center_y = (M["m01"]/M["m00"]) #得到轮廓的中心点
            center.append([center_x,center_y])
            valid_contours.append(contour)   #将第一次筛选后的轮廓存入新列表
    return center, valid_contours

def isParallel(contours: list , center: list , image: np.ndarray) -> np.ndarray :  
    """Function summary : Sift through and match the contours on both sides of the armor,
                          then mark the armor plate with a green dot
    
    Args:
        contours(list) : contours on image.
        center(list) : center of each contour.
        image(np.ndarray) : image with contours.
        
    Returns:
        np.ndarray : image with green dot on armor.
    """
    for i in range(len(contours)) :  
        for j in range(i + 1, len(contours)) :  
                rect_i = cv2.minAreaRect(contours[i])  
                rect_j = cv2.minAreaRect(contours[j])  
                ratio_i1 = rect_i[1][1]/rect_i[1][0]###rect_i[1][0] is width,rect[1][1] is height
                ratio_j1 = rect_j[1][1]/rect_j[1][0]
                if (1/ratio_i1)>1.8 and (1/ratio_j1)>1.8 and rect_i[1][0]>=12 and rect_j[1][0]>=12 and 1.3>rect_i[1][0]/rect_j[1][0]>0.8 :
                    ##width>height，此时长边是width，短边是height,将细长的轮廓筛出，轮廓长边大于12，装甲板两侧轮廓长边之比在0.8~1.3之间 
                    angle_diff = abs(rect_i[2] - rect_j[2])

                    if  angle_diff<8.5 and rect_i[2]<-45 and rect_j[2]<-45 and abs(center[i][1] - center[j][1])<int(rect_j[1][0]+rect_i[1][0])*0.2 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][0])*2 :
                        #轮廓应向上倾斜至少45度，相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), 10)

                if ratio_i1>1.8 and ratio_j1>1.8 and rect_i[1][1]>=12 and rect_j[1][1]>=12 and 1.3>rect_i[1][1]/rect_j[1][1]>0.8 :
                    #height>width，长边height，短边width,将细长的轮廓筛出，轮廓长边大于12，装甲板两侧轮廓长边之比在0.8~1.3之间
                    angle_diff = abs(rect_i[2] - rect_j[2])

                    if angle_diff<8.5 and rect_i[2]>-45 and rect_j[2]>-45 and abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][1])*0.2 and abs(center[i][0] - center[j][0])<int(rect_i[1][1]+rect_j[1][1])*2:
                       #轮廓应向上倾斜至少45度，相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), 10)

                if (rect_i[2]>-13 and rect_j[2]<-80) and (ratio_i1)>1.8 and (1/ratio_j1)>1.8 and rect_i[1][1]>=10 and rect_j[1][0]>=10 : 
                    #灯条竖直时，有时候轮廓会识别错误，此时装甲板两侧的轮廓一条向左倾斜很小的角度，另外一条向右倾斜很小角度

                    if abs(center[i][1] - center[j][1])<int(rect_i[1][1]+rect_j[1][0])*0.2 and abs(center[i][0] - center[j][0])<int(rect_i[1][1]+rect_j[1][0])*2:
                        #相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), 10)

                if (rect_j[2]>-13 and rect_i[2]<-80) and (ratio_j1)>1.8 and (1/ratio_i1>1.8) and rect_j[1][1]>=10 and rect_i[1][0]>=10 :
                    #灯条竖直时，有时候轮廓会识别错误，此时装甲板两侧的轮廓一条向左倾斜很小的角度，另外一条向右倾斜很小角度

                    if abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][0])*0.2 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][1])*2:  
                        #相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), 10)
                if (rect_j[2]==rect_i[2]==-90) and (1/ratio_j1)>1.8 and (1/ratio_i1>1.8) and rect_j[1][0]>=10 and rect_i[1][0]>=10:
                    #两个灯条均垂直时
                    if abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][0])*0.2 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][1])*2:   
                        #相邻轮廓中心点纵坐标之差小于灯条长度之和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), 10)
    return image 
def imageFilter(image: np.ndarray , Color: str) -> np.ndarray :
    """Function summary : process each frame by the function above
    
    Args:
        image(np.ndarray) : raw image.
        
    Returns:
        np.ndarray : image with green dot on armor.
    """
    image, contours = contourDrawing(image, Color)
    if contours is None or len(contours) == 0:  
        return image    
    center = []
    valid_contours = []
    center, valid_contours = getCenter(contours)
    image = isParallel(valid_contours, center,image)
    return image

if __name__ == "__main__" :
    video_capture = cv2.VideoCapture('test.mp4')  #打开视频，名字test.mp4
    Color = input("red or blue?:")
    if not video_capture.isOpened():  
        print("Error: Could not open video.")  
        exit()    
    while True:  
        ret, frame = video_capture.read()  
        if not ret:  
            break  
        img = imageFilter(frame , Color) 
        cv2.imshow("img",img)        
        if cv2.waitKey(1) == ord('q'):  
            break                          
    video_capture.release()  
    cv2.destroyAllWindows()
    cv2.waitKey()

    



