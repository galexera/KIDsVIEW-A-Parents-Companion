from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os


app = Flask(__name__)

#Child Maltreatment Detection
MODEL_ARCHITECTURE = 'E:\\PROJECTS\\KIDsVIEW\\TE_Project\\models\\model_CM.json'
MODEL_WEIGHTS = 'E:\\PROJECTS\\KIDsVIEW\\TE_Project\\models\\model_child_maltreatment_final.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


# Get weights into the model
model=load_model(MODEL_WEIGHTS)

def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict_classes(xtest_image)
    return preds

#Child Activity Recognition
MODEL_ARCHITECTUREE = 'E:\PROJECTS\KIDsVIEW\TE_Project\models\model_activity.json'
MODEL_WEIGHTSS = 'E:\PROJECTS\KIDsVIEW\TE_Project\models\model_activity_final.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTUREE)
loaded_model_json = json_file.read()
json_file.close()
modell = model_from_json(loaded_model_json)


# Get weights into the model
modell=load_model(MODEL_WEIGHTSS)
print('Model loaded. Check http://127.0.0.1:5000/')

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


model1 = "E:\PROJECTS\KIDsVIEW\TE_Project\pose_deploy_linevec_faster_4_stages.prototxt"
model2 = "E:\PROJECTS\KIDsVIEW\TE_Project\pose_iter_160000.caffemodel"

numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],[1,14],
               [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

modelo = cv2.dnn.readNetFromCaffe(model1, model2)

cor_ponto, cor_linha = (255, 128, 0), (7, 62, 248)

def keypoint_extract(path):
  print(path)
  print('kp')
  imagem = cv2.imread(path)
  cv2.imshow('img',imagem)
  imagem_copy = np.copy(imagem)

  imagem_width = imagem.shape[1]
  imagem_height = imagem.shape[0]

  height = 368
  width = int((height / imagem_height) * imagem_width)

  blob = cv2.dnn.blobFromImage(imagem, 1.0 / 255, (width, height), (0, 0, 0), swapRB = False, crop = False)

  modelo.setInput(blob)
  final = modelo.forward()

  final_height = final.shape[2]
  final_width = final.shape[3]

  pontos = []
  limite = 0.1
  for i in range(numero_pontos):
    mapa_confianca = final[0, i, :, :]
    _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)
    #print(confianca)
    #print(ponto)
    
    x = (imagem_width * ponto[0]) / final_width
    y = (imagem_height * ponto[1] / final_height)
    
    if confianca > limite:
      cv2.circle(imagem_copy, (int(x), int(y)), 8, cor_ponto, thickness = -1, 
                lineType=cv2.FILLED)
      cv2.putText(imagem_copy, "{}".format(i), (int(x), int(y)), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, 
                  lineType=cv2.LINE_AA)
      pontos.append((int(x), int(y)))
    else:
      pontos.append((-1,-1))

  keypoint = []
  for x in range(0,15):
    for y in range(0,2):
      keypoint.append(pontos[x][y])

  return keypoint

def prediction(pred):
  if(pred == 0):
    act = 'Eating'
  elif(pred == 1):
    act = 'Reading'
  elif(pred == 2):
    act = 'Sleeping'
  elif(pred == 3):
    act = 'Watching'
  return act

def model_predictt(img_path,modell):
#   imagem = cv2.imread(img_path)
  kp = keypoint_extract(img_path)
  print(kp)
  print('mp')
  df = pd.DataFrame([kp])
  clf_y_pred = modell.predict_classes(df.iloc[0:1]) 
  print(prediction(clf_y_pred[0]))
  p=prediction(clf_y_pred[0])
  return p

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def AfterRegister():
    # Main page
    return render_template('index.html')

@app.route("/login")
def login():
  return render_template("login.html")

@app.route("/register")
def register():
  return render_template("register.html")

@app.route("/abuse")
def abuse():
  return render_template("detection.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        
        if preds == [0]:
            prediction = 'INAPPROPRIATE BEHAVIOUR'
        else:
            prediction = 'APPROPRIATE BEHAVIOUR'
                     
        return prediction
    return None

@app.route("/activity")
def activity():
  return render_template("detect.html")

@app.route('/prediction', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        

        # Make prediction
        preds = model_predictt(file_path,modell)
        print(preds)
        # if preds[0][0] == 0:
        #     prediction = 'Infected with Pneumonia'
        # else:
        #     prediction = 'Not infected'
        
        return preds
    return None

if __name__ == '__main__':
    app.run(debug=True)
    app.run('localhost')
