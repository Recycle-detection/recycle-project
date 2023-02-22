import os 
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input #resnet 사용할것.
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import shutil
def model1(modeldir,imgname):
  """
  1차 모델 로드 & 예측

  args
      modeldir : 모델 경로
      imgname : 이미지 파일 이름+확장자명

  return
      result1 : bounding box그림 파일
      path2 : 예측 그림 경로
  """

  # 모델 로드
  path=os.path.join(modeldir+'/ultralytics/runs/detect/train2/weights/',"best.pt")
  model=YOLO(path)
  print("model 준비 완료")

  # 모델 예측
  path1=os.path.join(modeldir+'/ultralytics/ultralytics/yolo/data/data/test/images/',imgname)
  im1=cv2.imread(path1)
  result1=model.predict(source=im1, save=True, save_txt=True) 
  print("model 예측 완료")

  #시각화
  path2=os.path.join(modeldir+'/runs/detect/predict/',"image0.jpg")
  image = Image.open(path2)
  img = np.array(image)
  plt.figure(figsize=(30,30))
  plt.imshow(img)
  plt.show()
  
  return result1,path2

def jpg_image_to_array(image_path, size):
    image = Image.open(image_path)      # open image
    image = image.resize((size, size))  #resize
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) #convert to int ndarray
    print(im_arr.shape)
    return im_arr

def read_all_file(path):
    output = os.listdir(path)
    file_list = []

    for i in output:
        if os.path.isdir(path+"/"+i): 
            file_list.extend(read_all_file(path+"/"+i)) 
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)
    return file_list

def model1to2(imgpath,imgpath2):
  """
  1차 모델 결과 이미지에서 bounding box별로 사진 추출 & 2차 모델 입력 데이터 전처리

  args
      imgpath : 1차 모델 결과 이미지 텐서
      imgpath2 : 1차 모델 결과 이미지 경로

  return
      plot : bounding box별 이미지 파일
      x : 2차 모델 입력 데이터
  """

  #사진 추출
  image = cv2.imread(imgpath2)
  result2=imgpath
  crops = []
  for i in range(len(result2[0])):
    cor = list(result2[0].boxes.boxes[i][:4])
    crop = image[int(cor[1]):int(cor[3]), int(cor[0]):int(cor[2])]
    crops.append(crop)

  #시각화
  import matplotlib.pyplot as plt
  plt.figure(figsize=(20,10))
  for i in range(len(result2[0])):
    img=cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.save(f"/content/gdrive/MyDrive/object-detection/ultralytics/e/{i}.jpg","JPEG") 
    plt.subplot(2,5,i+1)
    plt.imshow(img)
  plt.show()

  # 입력 데이터 전처리
  from tensorflow.keras.preprocessing import image
  data = read_all_file("e")
  x_data = []
  for d in data:
    img = image.load_img(d,target_size= (224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x_data.append(x)
  x = np.asarray(x_data)

  return x

def model2(modeldir,x):
  """
  bounding box별 사진으로 2차 모델링 (오염도 분석) 

  args
      modeldir : 모델 경로
      x : 1차 모델 입력 데이터

  print
      오염 여부 출력

  return
      result : 2차 모델 오염도 분석 결과 
  """

  # 2차 모델 예측
  reconstructed_model = tf.keras.models.load_model("my_cnn.h5")
  result=reconstructed_model.predict(x)
  for i in range(len(result)):
    if result[i][0]==1:
      print(f"{i+1}번째 사진에서 오염을 제거해주세요.")
    else:
      print(f"{i+1})번째 사진은 깨끗합니다. 분리배출해주세요.")
  
  #1차 예측 모델 결과 삭제
  dir_path = os.path.join(modeldir+'/','runs')
  shutil.rmtree(dir_path)
  return result

