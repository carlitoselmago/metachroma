import cv2
import mediapipe as mp
from classes.facecapture import facecapture
from threading import Thread

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


FC=facecapture(face_mesh)

fp = Thread(target=FC.faces_process_get)
fp.start()


#results=FC.get_crop_from_faces(img_rgb,face_mesh)


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

                for i in eye_points:
                    x = int(face_landmarks.landmark[i].x * img_rgb.shape[1])
                    y = int(face_landmarks.landmark[i].y * img_rgb.shape[0])
                    x_values.append(x)
                    y_values.append(y)

                all_x_values.extend(x_values)
                all_y_values.extend(y_values)

            x_min, x_max, y_min, y_max = min(all_x_values), max(all_x_values), min(all_y_values), max(all_y_values)
            w, h = x_max - x_min, y_max - y_min

            cx, cy = x_min + w // 2, y_min + h // 2
            b_dim = max(w, h) * zoom_scale
            bx, by = max(0, cx - b_dim // 2), max(0, cy - b_dim // 2)
        else:
            bx = 0
            by = 0
            b_dim = max(img_rgb.shape[1], img_rgb.shape[0])

        if transition_frames > 0:
            frame_count = (frame_count + 1) % transition_frames
            bx = int((bbox_curr[0] * (transition_frames - frame_count) + bx * frame_count) / transition_frames)
            by = int((bbox_curr[1] * (transition_frames - frame_count) + by * frame_count) / transition_frames)
            b_dim = int((bbox_curr[2] * (transition_frames - frame_count) + b_dim * frame_count) / transition_frames)

        bbox_curr = [bx, by, b_dim]

        img_cropped = img_rgb[by:by+b_dim, bx:bx+b_dim]
        img_resized = cv2.resize(img_cropped, (width, height))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

        ###
        # aspect ratio of original image
        aspratio_crop = img_cropped.shape[1] / img_cropped.shape[0]
        # aspect ratio of desired size
        aspratio_screen = width / height

        if(aspratio_crop < aspratio_screen):
            # resize width to fit
            wtemp = width
            htemp = int(width / aspratio_crop)
        else:
            htemp = height
            wtemp = int(height * aspratio_crop)

        img_temp = cv2.resize(img_cropped, (wtemp, htemp))

        # Cropping the image
        startRow = int(max(0,int((img_temp.shape[0]-height)/2)))
        startCol = int(max(0,int((img_temp.shape[1]-width)/2)))
        img_resized = img_temp[startRow:(startRow+height), startCol:(startCol+width)]

        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
        ###
        cv2.imshow("img", img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
