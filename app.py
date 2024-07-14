from __future__ import division, print_function
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model22 = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
app = Flask(__name__, static_url_path='')


app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('out.h5')
print('Model loaded. Start serving...')


classes = ['Glioma','Meningioma','NoTumor','Pituitary']


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    print(img_path)
    
    imgp = load_img(img_path, target_size = (180,180,3))
    imgp = img_to_array(imgp)
    imgp = imgp/255
    imgp = np.expand_dims(imgp,axis=0)
    p=np.argmax(model.predict(imgp,verbose=0))



    img = cv2.imread(img_path)
    img = cv2.resize(img,(640, 640))
    
    result = model22(img)
    data_frame = result.pandas().xyxy[0]
    
    indexes = data_frame.index
    for index in indexes:
        x1 = int(data_frame['xmin'][index])
        y1 = int(data_frame['ymin'][index])
        x2 = int(data_frame['xmax'][index])
        y2 = int(data_frame['ymax'][index ])
    
        label = classes[p]
        conf = data_frame['confidence'][index]
        text = label + ' ' + str(conf.round(decimals= 2))
    
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(img, text, (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255,255,0), 2)
    cv2.imwrite('./uploads/img.jpg',img)
    
    
    return classes[p]


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        pred = model_predict(file_path, model)
        
        fname="./img.jpg"
            
        
    return render_template('predict.html', outfile=fname,result=pred)



if __name__ == '__main__':
        app.run()

