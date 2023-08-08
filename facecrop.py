import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

def crop_centered(img, final_resolution):
    height, width = img.shape[:2]

    left = (width - final_resolution[0]) // 2
    right = left + final_resolution[0]
    top = (height - final_resolution[1]) // 2
    bottom = top + final_resolution[1]

    img = img[top:bottom, left:right]

    return img

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=10,
                                  min_detection_confidence=0.8,
                                  min_tracking_confidence=0.8)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
width = 3840#1920
height = 2160#1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
zoom_scale = 2  # decrease this value to decrease the margin
transition_frames = 220  # You can modify transition_frames as needed
frame_count = 0

bbox_curr = [0, 0, cap.get(cv2.CAP_PROP_FRAME_WIDTH),
             cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]   # Initial bounding box
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    success, img = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    for (x, y, w, h) in faces: 
        pass
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces) > 0: 

        all_x_values = [x, x + w]
        all_y_values = [y, y + h]

        x_min, x_max, y_min, y_max = min(all_x_values), max(all_x_values), min(all_y_values), max(all_y_values)
        w, h = x_max - x_min, y_max - y_min

        cx, cy = x_min + w // 2, y_min + h // 2
        b_dim = max(w, h) * zoom_scale
        bx, by = max(0, cx - b_dim // 2), max(0, cy - b_dim // 2)
    else:
        bx = 0
        by = 0
        b_dim = max(img.shape[1], img.shape[0])

    if transition_frames > 0:
        frame_count = (frame_count + 1) % transition_frames
        bx = int((bbox_curr[0] * (transition_frames - frame_count) + bx * frame_count) / transition_frames)
        by = int((bbox_curr[1] * (transition_frames - frame_count) + by * frame_count) / transition_frames)
        b_dim = int((bbox_curr[2] * (transition_frames - frame_count) + b_dim * frame_count) / transition_frames)

    bbox_curr = [bx, by, b_dim]

    #cv2.rectangle(img, (bx, by), (bx+b_dim, by+b_dim), (0,255,0), 2)
    #cv2.imshow('Bounding Box', img)
    img_cropped = img[by:by+b_dim, bx:bx+b_dim]
    img_resized = cv2.resize(img_cropped, (1920, 1080))
    cv2.imshow("Image", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()