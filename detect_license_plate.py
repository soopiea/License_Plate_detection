#!/usr/bin/python3

import os
from ultralytics import YOLO
from PIL import Image
import itertools
import cv2
import easyocr
import pandas as pd

def main():
  #creating the dataframe
  columns = ['file_location','license_plate','confidence_score']
  df = pd.DataFrame(columns=columns)

  for i, r in enumerate(results):
    image_path = r.path
    print(f'Image file: {image_path}')
    xyxy = r.boxes.xyxy
    x_conf =r.boxes.conf

    #Appending the confidence score with the bounding box points
    if xyxy.device.type == 'cuda':
      data_points = list(itertools.chain(*xyxy.cpu().numpy()))
      confidence_score = x_conf.cpu().numpy()[0]
    else:
      data_points = list(itertools.chain(*xyxy.numpy()))
      confidence_score = x_conf.numpy()[0]

    #Only read license plate with confidense score of greater than or equal 0.5
    if confidence_score >= 0.5:

      #Get the xywh  to feed to opencv bounding box
      x1, y1, x2, y2 = data_points
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      w, h = x2-x1, y2-y1

      #Crop the image to the yolov8 predicted bouding box
      img = cv2.imread(image_path)
      crop_img = img[y1:y1+h, x1:x1+w]

      #Read the license plate number from crop iamge
      reader = easyocr.Reader(['en'])
      result = reader.readtext(crop_img)

      #Will only get the first result from the result list(for improvement)
      if len(result) == 0: 
        print("Cant detect/read license plate in the image")
        continue
      _, text, char_prob = result[0]
      print(f'Text: {text}, Probability: {char_prob}')

      # append to df
      df.loc[len(df)] = [image_path, text, char_prob]

  #Save the csv file
  save_path = f"{project_path}/results/license_plate_detection_result.csv"
  df.to_csv( save_path, index=False)
  print(f'File saved to : {save_path}')

if __name__ == '__main__':
  project_path = os.getcwd()
  os.chdir(project_path)

  #Process the images in the test dataset
  images_path = os.path.join(project_path,'run_test_images')
  image_list = os.listdir(images_path)
  image_list = list(map(lambda x: f'{images_path}/'+ x, image_list))

  #Retrieve the model
  model = YOLO("dataset/runs/detect/train3/weights/best.pt")
  results = model(image_list)

  main()