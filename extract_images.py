import json
import os
import cv2

with open("./datasets/annotations.json","r") as f:
    file_name = json.load(f)

for i in file_name['images']:
    file_path = os.path.join("./datasets/images/", i)
    image = cv2.imread(file_path)
    file_write = os.path.join("./datasets/challenge/8900/", i)
    cv2.imwrite(file_write, image)