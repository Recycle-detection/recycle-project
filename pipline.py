import os 
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def model1(modeldir,imgname):
    """
    모델 로드 & 예측

    args
        modeldir : 모델 경로
        imgname : 이미지 파일 이름+확장자명

    return
        plot : bounding box그림
    """

    # 모델 로드
    path=os.path.join(modeldir+'/ultralytics/runs/detect/train2/weights/',"best.pt")
    model=YOLO(path)
    print("model 준비 완료")

    # 모델 예측
    path1=os.path.join(modeldir+'/ultralytics/ultralytics/yolo/data/data/test/images/',imgname)
    im1=cv2.imread(path1)
    model.predict(source=im1, save=True, save_txt=True) 
    print("model 예측 완료")

    #시각화
    path2=os.path.join(modeldir+'/ultralytics/runs/detect/predict9/',"image0.jpg")
    image = Image.open(path2)
    img = np.array(image)
    plt.figure(figsize=(30,30))
    plt.imshow(img)
    plt.show()