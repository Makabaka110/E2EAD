import argparse
import base64
import json
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

import torch
import torchvision.transforms as transforms
from networks import *


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None



def preprocess_image(image):

    # Resize the image to match the input size of the model
    image = cv2.resize(image, (128, 64))

    # Add a batch dimension and convert the image to a PyTorch tensor
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)

    return image


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Get the current steering angle and throttle from the simulator
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        # Get the current image from the simulator and preprocess it
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = preprocess_image(image)
        image = image.to(device).float()
        # Pass the image through the model to get a steering angle prediction
        with torch.no_grad():
            model.eval()
            steering_angle = model(image).item()

        # The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = 0.2
        # Send the steering angle and throttle back to the simulator
        print('Steering angle =', '%5.2f'%(float(steering_angle)), 'Throttle =', '%.2f'%(float(throttle)), 'Speed  =', '%.2f'%(float(speed)))
        send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("Connected ",sid)
    send_control(3, 20)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)
    print("send control")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    # Load the model from the specified path
    with open(args.model, 'r') as f:
        model = SteeringModel()
        # Use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
        model.load_state_dict(torch.load('model.h5'))
        model.eval()

    # Start the server and listen for incoming connections
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)