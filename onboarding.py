import cv2
import math
import face_detection
from app import db, Student


video_capture = cv2.VideoCapture(0)
counter = 8

while True:
    _, frame = video_capture.read()
    frame, face_box, face_coords = face_detection.detect_faces(frame)
    text = 'Image will be taken in {}..'.format(math.ceil(counter))
    if face_box is not None:
        frame = face_detection.write_on_frame(frame, text, face_coords[0], face_coords[1]-10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    counter -= 0.1
    if counter <= 0:
        #imgpath = 'static/AttendanceUploads/true_image_' + student_id + '.png'
        imgpath = 'static/AttendanceUploads/true_image_2.png'
        cv2.imwrite(imgpath, face_box)
        #Add to DB HOW?
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Onboarding Image Captured")