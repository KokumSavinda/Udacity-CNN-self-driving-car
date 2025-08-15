import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import cv2
import torch
from model1 import CNNModel1  # your model definition file

# Initialize Flask + SocketIO server
sio = socketio.Server()
app = Flask(__name__)
model = CNNModel1()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('Model1Run1.pth', map_location=device))
model.eval()
model.to(device)

def preprocess_image(img):
    img = np.asarray(img)
    img = img[60:-25, :, :]               # Crop
    img = cv2.resize(img, (200, 66))      # Resize
    img = img / 127.5 - 1.0               # Normalize
    img = np.transpose(img, (2, 0, 1))    # Channel-first for PyTorch
    img = np.expand_dims(img, axis=0)     # Add batch dim
    return torch.from_numpy(img).float().to(device)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Get image from simulator
        img_str = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_str)))
        image_tensor = preprocess_image(image)

        # Predict steering angle
        with torch.no_grad():
            steering_angle = float(model(image_tensor).cpu().item())

        print(f"Predicted steering angle: {steering_angle:.4f}")

        # Send back prediction and throttle
        send_control(steering_angle,0.9)

@sio.on('connect')
def connect(sid, environ):
    print("Simulator connected.")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
