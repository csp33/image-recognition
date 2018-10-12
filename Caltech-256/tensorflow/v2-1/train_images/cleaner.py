import os
import glob
import cv2

image_size=128

classes=os.listdir('.')
print(classes)
for fields in classes:
    path = os.path.join(".", fields, '*g')
    files=glob.glob(path)
    #Files contains ALL THE FILES
    for fl in files:
        try:
            image=cv2.imread(fl)
            cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        except:
            print("File " + fl + " broken.")
            os.remove(fl)
            continue
