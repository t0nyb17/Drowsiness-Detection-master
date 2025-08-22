import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame  # Import pygame for the alarm

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor model
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("Error: shape_predictor_68_face_landmarks.dat not found!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Status tracking
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to detect blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Open eyes
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Closed eyes

# Main loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    face_detected = False  # <-- track whether a face is detected

    for face in faces:
        face_detected = True
        face_frame = frame.copy()

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Detect blinking
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41],
                             landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47],
                              landmarks[46], landmarks[45])

        # Update status
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                # Play alarm sound when sleeping
                pygame.mixer.music.load('warning-sound-6686.mp3')  # Ensure this path is correct
                pygame.mixer.music.play()
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                # Play alarm sound when drowsy
                pygame.mixer.music.load('warning-sound-6686.mp3')  # Ensure this path is correct
                pygame.mixer.music.play()
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)

    if face_detected: 
        cv2.imshow("Result of Detector", face_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
