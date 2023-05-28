from ultralytics import YOLO

import numpy as np


model = YOLO('./runs/classify/train/weights/best.pt')  # load a custom model

results = model('D:/Documents/Cristobal Paraon DOCS/3rd YEAR/2ndSem/Capstone/Pc_Based/code/copra_dataset/train/classC/IMG_20230418_091114.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])