# Import necessary libraries and modules
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import random
import imutils
import sys

from flask import Flask, send_file, Response

IMG_CATEGORIES = {0:'Safe driving', 
                1:'Texting', 
                2:'Operating radio', 
                3:'Drinking', 
                4:'Reaching behind'}

# Specifying the size of image, same as used for training
IMG_SIZE = 299

# Load the model for predictions
pred_model = load_model('Best_Classification_Model.hdf5')

# Specifying font to be used 
cv2Font = cv2.FONT_HERSHEY_SIMPLEX

# Set the default prediction
current_prediction = 'Safe driving'

app = Flask(__name__)
@app.route('/ping')
def ping():
    return "API working"

@app.route('/test', methods=["GET"])
def predict_test_image():
    frame = cv2.imread('test_img.jpg', cv2.IMREAD_UNCHANGED)

    # get the predictions for a frame
    predicted_frame = predict_frame(frame)

    # save the output image
    filename = 'output.jpg'
    cv2.imwrite(filename, predicted_frame)

    # return response
    _, jpeg = cv2.imencode('.jpg', predicted_frame)
    frame_bytes = jpeg.tobytes()
    return Response(frame_bytes, mimetype='image/jpg')

@app.route('/predict', methods=["GET"])
def predict_feed():
    vidObj = cv2.VideoCapture(0) #0 = webcam feed
    return Response(predict_video(vidObj), mimetype='multipart/x-mixed-replace; boundary=frame')

def predict_video(vidObj):
    success = 1
    while True:
        # read a frame from the video feed
        success, frame = vidObj.read()

        # get the predictions for a frame
        predicted_frame = predict_frame(frame)
        _, jpeg = cv2.imencode('.jpg', predicted_frame)
        frame = jpeg.tobytes()

        # use yield (and NOT return) so that predictions are returned continuously until the feed stops
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def predict_frame(frame):

    # preprocess the image frame
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # convert the image to the size as required by the model
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype("float16")
    
    # predict
    prediction = pred_model.predict(np.expand_dims(new_array, axis=0))
    result = np.argmax(prediction, axis=1)
    current_prediction = IMG_CATEGORIES.get(int(result))
    # print(current_prediction)

    if result == 0:
        #Safe driving, text displayed in GREEN
        cv2.putText(frame, "Driver's Status: "+ current_prediction, (50,50), cv2Font, 0.7, (0,255,0), 2)

    else:
        #Unsafe driving, text displayed in RED
        cv2.putText(frame, "Driver's Status: "+ current_prediction, (50,50), cv2Font, 0.7, (0,0,255), 2)

    return frame


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, threaded=True)