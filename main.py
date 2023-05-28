from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='D:/Documents/Cristobal Paraon DOCS/3rd YEAR/2ndSem/Capstone/Pc_Based/code/copra_dataset',
            epochs=10, imgsz=64)