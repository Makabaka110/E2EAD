import base64
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import torch
import torchvision.transforms as transforms


from networks import *
from dataloader import *
import config


sio = socketio.Server()
app = Flask(__name__)
model = None




def preprocess_image(image):
    
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    # image = image.float()  # Convert to float type

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
        # image = image.to(device).float()
        # Pass the image through the model to get a steering angle prediction
        try:
            with torch.no_grad():
                steering_angle = model(image)
                print('steering_angle', steering_angle)
                steering_angle = steering_angle.item()
                print('steering_angle.item()', steering_angle)

            # The driving model currently just outputs a constant throttle. Feel free to edit this.
            if (float(speed) < 10):
                throttle = 0.4 
            else:
                # When speed is below 20 then increase throttle by speed_factor
                if ((float(speed)) < 25):
                    speed_factor = 1.35
                else:
                    speed_factor = 1.0 
                if (abs(steering_angle) < 0.1): 
                    throttle = 0.1 * speed_factor
                elif (abs(steering_angle) < 0.5):
                    throttle = 0.05 * speed_factor
                else:
                    throttle = 0.01 * speed_factor
            # Send the steering angle and throttle back to the simulator
            # print('Steering angle =', '%5.2f'%(float(steering_angle)), 'Throttle =', '%.2f'%(float(throttle)), 'Speed  =', '%.2f'%(float(speed)))
            print('Steering angle =', '%5.2f'%(float(steering_angle)), 'Throttle =', '%.2f'%(float(throttle)))
            # send_control(steering_angle, throttle, speed)
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    
    print("Connected ",sid)
    send_control(0.0,0.0)


# def send_control(steering_angle, throttle, speed):
def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
        # 'speed': speed.__str__()
    }, skip_sid=True)
    # print("send control")

if __name__ == '__main__':
    model = config.MODEL_TYPE_CLASS()
    # # Use CUDA if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model.to(device)
    model.load_state_dict(torch.load('models/Keyboard_10mins/ResNet50/epho_9.pth'))
    model.eval()

    # Start the server and listen for incoming connections
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)