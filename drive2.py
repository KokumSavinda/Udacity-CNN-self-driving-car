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
from model2 import CNNModel2  # ✅ Make sure this class is in model.py

# Initialize Flask + SocketIO server
sio = socketio.Server()
app = Flask(__name__)
model = CNNModel2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('Model2Run1.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing function
def preprocess_image(img):
    img = np.asarray(img)
    img = img[60:-25, :, :]               # Crop sky and hood
    img = cv2.resize(img, (200, 66))      # Resize to model input
    img = img / 127.5 - 1.0               # Normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))    # Channel-first for PyTorch
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    return torch.from_numpy(img).float().to(device)

# Event handler: when simulator sends telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Decode incoming image from simulator
        img_str = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_str)))
        image_tensor = preprocess_image(image)

        # Predict both steering and throttle
        with torch.no_grad():
            output = model(image_tensor)  # shape: [1, 2]
            steering_angle = float(output[0, 0].item())
            throttle = float(output[0, 1].item())

        print(f"Predicted | Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}")
        send_control(steering_angle, throttle)

# Event handler: simulator connects
@sio.on('connect')
def connect(sid, environ):
    print("Simulator connected.")
    send_control(0, 0)

# Function to send control commands back to simulator
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )

# Start the server
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
