#!/usr/bin/env python

import os
import argparse
import chainer
import numpy as np
import re, base64, cv2
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from train_mnist import MLP

hostname = os.environ['HOSTNAME']
app = Flask(__name__)
CORS(app) # local post by Ajax
model = MLP(100, 10)
chainer.serializers.load_npz('result/pretrained_model', model)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ans = get_answer(request)
        return jsonify({'ans': ans[1],
           'c0': '{:.4f}'.format(ans[0][0]), 'c1': '{:.4f}'.format(ans[0][1]),
           'c2': '{:.4f}'.format(ans[0][2]), 'c3': '{:.4f}'.format(ans[0][3]),
           'c4': '{:.4f}'.format(ans[0][4]), 'c5': '{:.4f}'.format(ans[0][5]),
           'c6': '{:.4f}'.format(ans[0][6]), 'c7': '{:.4f}'.format(ans[0][7]),
           'c8': '{:.4f}'.format(ans[0][8]), 'c9': '{:.4f}'.format(ans[0][9]),
           'hostname': hostname})
    else:
        return render_template('index.html')

def get_answer(req):
    img_str = re.search(r'base64,(.*)', req.form['img']).group(1)
    nparr = np.fromstring(base64.b64decode(img_str), np.uint8)
    img_src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_neg = 255 - img_src
    img_gray = cv2.cvtColor(img_neg, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray,(28,28))
    cv2.imwrite("images/{}.jpg".format(datetime.now().strftime('%s')), img_resize)
    data = img_resize.astype(np.float32)
    ans = single_predictor(model, data)
    return ans

def single_predictor(model, image):
    test = np.array(image).reshape(1, -1)
    pred = model(test)
    pred = chainer.functions.softmax(pred).data
    label_y = [np.argmax(pred[i]) for i in range(len(pred))]
    return (pred[0], label_y[0])


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000) 
