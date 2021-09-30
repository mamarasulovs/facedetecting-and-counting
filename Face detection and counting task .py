# Project face detecting and counting by SardorM

#we should import some necessary packages : opencv and mediapipe face detection model

import cv2
import mediapipe as mp

# we define mediapipe Face detector

face_detection = mp.solutions.face_detection.FaceDetection(0.6)

img = cv2.imread ("image.jpg")


# Detection function

def detector(frame):

    count = 0
    height, width, channel = frame.shape

    # Convert frame BGR to RGB colorspace

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#imgRGB = cv2.cvtColor('frame')
    # Detect results from the frame
    result = face_detection.process(imgRGB)

    # print(result.detections)

    # Extract information from the result

    # If some detection is available then extract those information from result
    try:
        for count, detection in enumerate(result.detections):

            # print(detection)

        
            # Extract Score and bounding box information 

            score = detection.score
            box = detection.location_data.relative_bounding_box

            # print(score)
            # print(box)

            x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width), int(box.height*height)
            score = str(round(score[0]*100, 2))

            print(x, y, w, h)

            # Draw rectangles

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y-25), (0, 0, 255), -1)

            cv2.putText(frame, score, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        count += 1
        print("Found ",count, "Faces!")

        
    # If detection is not available then pass 
    except:
        pass

    return count, frame


# Detect from a image file

# run the detection
# count, output = detector(img)

# cv2.putText(output, "Number of Faces: "+str(count),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )

# cv2.imshow("output", output)
# cv2.waitKey(0)


# Detect from a video file
# load the video file  or rtsp camera adress:
#cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture('rtsp://admin:12345678a@195.158.6.26:555/cam/realmonitor?channel=1&subtype=0')
#cap = cv2.VideoCapture(0)

'''def make_1080p():
	cap.set(3, 1920)
	cap.set(4, 1080)
	
def make_720p():
	cap.set(3, 1280)
	cap.set(4, 720)
	
def change_res(width,height):
	cap.set(3,width)
	cap.set(4,height)
	
#make_1080p()

change_res(4000,2000)
#cap.set(3,640)
#cap.set(4,320) '''
'''def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent=75)
	#count, output = detector(frame)
	#cv2.putText(output, "Number of Faces: "+str(count),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )
    cv2.imshow('frame75', frame75)
    frame150 = rescale_frame(frame, percent=150)
    cv2.imshow('frame150', frame150)
'''
while True:
    _, frame = cap.read()
    count, output = detector(frame)
    cv2.putText(output, "Number of Faces: "+str(count),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )
    cv2.imshow('frame', output)
    if cv2.waitKey(15) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()