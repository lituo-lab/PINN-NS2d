import os
import cv2

file_dir = 'point256/'
list = []
for root ,dirs, files in os.walk(file_dir):
    for file in files:
        list.append(file)      # 获取目录下文件名列表


video = cv2.VideoWriter('point256.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(4800,3200))

for i in range(0,len(list)):
    #读取图片
    img = cv2.imread('point256/epoch-'+str(i*200)+'.png')
    video.write(img)

# 释放资源
video.release()

