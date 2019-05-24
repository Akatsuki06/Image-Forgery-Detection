from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from keras import backend as K
from cnn.lib.data import Data
from cnn.lib.model import CNNModel
import tensorflow as tf
import time
import os

model = CNNModel()
model.load('cnn/model/model.h5')
graph = tf.get_default_graph()

def get_predictions(name,img_url):
    global graph
    with graph.as_default():
        data = Data()
        img = data.import_image(img_url)
        result = model.predict(img).flatten()
        print(result)
        d= {
        'real': "{:.2f}".format(result[0]*100),
        'fake': "{:.2f}".format(result[1]*100)
        }
    return d['real'],d['fake']




def simple_upload(request):
    if request.method == 'POST':
        myfile = request.FILES.get('file',None)
        if not myfile:
            return render(request, 'index.html')

        fs = FileSystemStorage()
        fname,extension = os.path.splitext(myfile.name)
        # rl = fname[]
        name = str(int(time.time()))
        filename = fs.save('{0}{1}'.format(name,extension),myfile)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url)
        real,fake = get_predictions(fname,uploaded_file_url[1:])
        result = 'FAKE'
        if real>fake:
            result='REAL'
        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url,
            'real': real,
            'fake': fake,
            'result':result
        })
    return render(request, 'index.html')
