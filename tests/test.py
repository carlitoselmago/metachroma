from face_detection import RetinaFace
import cv2



cap = cv2.VideoCapture(1)
width = 1920
height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    success, img = cap.read()

    # 0 means using GPU with id 0 for inference
    # default -1: means using cpu for inference
    detector = RetinaFace(gpu_id=0) 
    all_faces = detector([img,img])
    box, landmarks, score = all_faces[0][0]
    print(box)

    #cv2.rectangle(img, (bx, by), (bx+b_dim, by+b_dim), (0,255,0), 2)
    #cv2.imshow('Bounding Box', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()