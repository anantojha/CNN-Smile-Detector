# ------------------------------------------------------------------------------------------
#   COMP 3106 - Introduction to Artificial Intelligence
#   Term Project:   Smile Detection
#   Description:    Script to detect smiles via webcam feed and trained model
# ------------------------------------------------------------------------------------------
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required=True, 
    help='path to cascade')
ap.add_argument('-m', '--model', required=True, 
    help='path to model')
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])

print('[INFO] starting video capture...')
camera = cv2.VideoCapture(0)

# video loop
while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)

    if args.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (faceX, faceY, faceWidth, faceHeight) in rects:
        roi = gray[faceY:faceY + faceHeight, faceX:faceX + faceWidth]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities - predict
        (notSmiling, Smiling) = model.predict(roi)[0]
        label = 'Smiling' if Smiling > notSmiling else "Not Smiling"

        # display the box and label
        if label == 'Smiling':
            cv2.putText(frameClone, label, (faceX, faceY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frameClone, (faceX, faceY), (faceX + faceWidth, faceY + faceHeight), (0, 255, 0), 2)
        else:
            cv2.putText(frameClone, label, (faceX, faceY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (faceX, faceY), (faceX + faceWidth, faceY + faceHeight), (0, 0, 255), 2)

    # show our detected face along with smiling/not smiling labels
    cv2.imshow('Face', frameClone)

    # if 'q' key is pressed, stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
