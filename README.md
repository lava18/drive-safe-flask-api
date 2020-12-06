# drive-safe-docker
A Flask API to predict the driver's state (`Safe Driving`, `Texting`, `Operating radio`, `Drinking`, `Reaching behind`) on a live webcam feed. 

## Steps
1. Clone this repo using `https://github.com/lava18/drive-safe-flask-api.git`
2. Create a virtual environment on your machine (with `pip` or `conda`)
3. Activate the virtual environment and install the dependencies: \
`pip install -r requirements.txt`
4. Start the Flask server: \
 `python app.py`\
This will start a server on port 4000.
5. Go to the browser and test your API: \
`http://localhost:4000/ping`
6. Now, invoke your API for a test image:\
`http://localhost:4000/test`
7. Finally, get predictions on your live webcam stream:
`http://localhost:4000/predict`
This will switch on your webcam and start streaming the video to the model.
(On Mac, you may need to give the Terminal the permission to access the web camera.)
You will be able to view the predictions in real-time on your browser.

## References
This repo uses a deep learning Keras model trained on [this](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) Kaggle dataset. 