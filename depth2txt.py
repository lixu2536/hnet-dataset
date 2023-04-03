import cv2
import numpy as np

### This is an example of write the 3D .txt file from UNet.
# model = load_model(dir + '/model_UNet_fringe.h5', custom_objects={'rmse' : rmse})
# Z_1= model.predict(X_test[1:2], batch_size =1, verbose =1)

# file = './point cloud/00f32.tiff'
file = './point cloud/new-model-ssim50.png'
# file = './point cloud/e200.png'

Z = cv2.imread(file, cv2.IMREAD_ANYDEPTH)

filepath = './point cloud/new-model-ssim50.txt'
output = open(filepath, 'w')

for i in range(640):
   for j in range(480):
       if (Z[j,i]>50):
           output.write(str(i))
           output.write(" ")
           output.write(str(480-1-j))
           output.write(" ")
           output.write('%.6f'%(Z[j,i]))
           output.write("\n")
output.close()