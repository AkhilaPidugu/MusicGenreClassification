from contextlib import nullcontext
import re
from werkzeug.utils import redirect, secure_filename
from logging import log
from types import MethodType
import flask as f
from flask import Flask, render_template
from flask.globals import request
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow import keras
import math
import librosa
from pathlib import Path



app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

saved_model=load_model('model')
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
saved_model.compile(optimizer=optimiser,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

def process_input(music_file,track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FFT=2048
    HOP_LENGTH=512
    TRACK_DURATION=track_duration 
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS=10

    samples_per_segment = int(SAMPLES_PER_TRACK/NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment/HOP_LENGTH)

    signal, sample_rate=librosa.load(music_file,sr=SAMPLE_RATE)
  
    for d in range(10):

        start = samples_per_segment *d
        finish = start+samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft= N_FFT,hop_length=HOP_LENGTH)
        mfcc =mfcc.T
        print(mfcc)
        return mfcc

genre_dict ={0:"Blues",1:"Classical",2:"Country",3:"Disco",4:"Hiphop",5:"Jazz",6:"Metal",7:"Pop",8:"Reggae",9:"Rock"}

@app.route('/')
def hello():
    return render_template('form.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    
    music_file = request.files['audioFileData']
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], music_file.filename)
    music_file.save(path)
        
    inputfile=process_input(path,30)
    x_to_predict = inputfile[np.newaxis, ...,np.newaxis]
    d = saved_model.predict(x_to_predict)
    predicted_index = np.argmax(d, axis=1)
    return render_template('predict.html')+"<h4>The Genre of music  is: </h4>"+genre_dict[int(predicted_index)]
    
if __name__ == '__main__':
    app.run(debug=True)