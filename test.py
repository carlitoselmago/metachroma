import dlib
import cv2

cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Capture video from the webcam
cap = cv2.VideoCapture(0)
width = 3840
height = 2160
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    # Read the frame
    ret, img = cap.read()

    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect the faces
    detections = cnn_face_detector(rgb_img, 1)
    
    # Draw the rectangle around each face
    for i, d in enumerate(detections):
        face_rect = d.rect
        x = face_rect.left()
        y = face_rect.top()
        w = face_rect.right() - x
        h = face_rect.bottom() - y

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()