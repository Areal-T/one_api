# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import time
 
from datetime import timedelta

vgg16 = models.vgg16(pretrained=True)
num_features = vgg16.classifier[3].in_features
vgg16.classifier[3] = nn.Linear(num_features, 2048)
in_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(in_features, 2048)
num_features = vgg16.classifier[6].out_features
vgg16.classifier[6] = nn.Linear(num_features, 7)

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

model = torch.load('./resnet_model.pth')
vgg16.load_state_dict(model)


# @app.route('/')
# def home():
#     return render_template('upload.html')
 
 
# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__) 
 
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
        
        image = Image.open(upload_path)

        transformer = transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        start_time = time.time()
        tensor = transformer(image).type(torch.float32).reshape(1,3,224,224)
        with torch.no_grad():
            outputs = vgg16(tensor)
            _,predicted = torch.max(outputs.data,1)

        if predicted.item()==0:
            result ='产品合格'
        elif predicted.item()==1:
            result ='图片不完整，无法判断'
        else:
            result ='产品不合格'
        
        end_time = time.time()

        time1 = end_time-start_time
        return render_template('upload_ok.html',userinput=user_input,prediction='图片识别为：{}'.format(result),val1='推理时间：{:.3f}s'.format(time1))
 
    return render_template('upload.html')

 
 
if __name__ == '__main__':
    # app.debug = True
    app.run(port=82)