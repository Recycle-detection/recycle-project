# recycle-project 설명

<br/>

---
## 데이터
- raw_images : 원본 데이터
- aug_data : 데이터 증강 데이터

---
## 모델링 파일
### 객체 탐지 1차 모델
- yolov7,yolov7_2 : yolov7 파일
- yolov8 : yolov8 이용 모델링 

### 오염여부 분석 2차 모델
- Detection 후 Crop: yolov8로 객체 탐지 후 자르기
- CNN: 오염여부 판단(이진분류)

<br/>

---
## 최종 파일
- ultralytics : yolov8 모듈
- pipline : 최종 파이프라인
- 실행 : pipline 실행 파일
