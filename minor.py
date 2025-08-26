import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import cv2 as cv
import numpy as np
import csv
from datetime import datetime
import time
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from sklearn.svm import SVC

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load(r"faces_embeddings_done_4classes.npz")
X, Y = faces_embeddings['arr_0'], faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier(r"haarcascade_frontalface_default (1).xml")
model = pickle.load(open(r"svm_model_160x160.pkl", 'rb'))

# Create a named window with full screen flag
cv.namedWindow('Face Recognition:', cv.WINDOW_FULLSCREEN)

# Open the video capture device (webcam)
cap = cv.VideoCapture(0)


# Set to track unique faces
unique_faces = set()

# CSV file setup
csv_file_path = r"detected_faces.csv"
csv_unique_faces_path = r"unique_faces.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time of Entry"])

with open(csv_unique_faces_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name"])

# Function to log face to CSV
def log_face(name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time])

def log_unique_face(name):
    with open(csv_unique_faces_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name])

def capture_new_face(name):
    captured_faces = []
    parent_dir = r"face-images"
    user_dir = os.path.join(parent_dir, name)

    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    for i in range(10):  # Capture 10 frames
        _, new_frame = cap.read()
        gray_new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        new_faces = haarcascade.detectMultiScale(gray_new_frame, 1.3, 5)

        for (x, y, w, h) in new_faces:
            face_img = new_frame[y:y+h, x:x+w]
            face_img = cv.resize(face_img, (160, 160))
            img_path = os.path.join(user_dir, f"{name}_{i+1}.jpg")
            cv.imwrite(img_path, face_img)  # Save the captured face image
            captured_faces.append(face_img)
            cv.rectangle(new_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv.putText(new_frame, f"Capturing {i+1}/10", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv.imshow("Capturing new face", new_frame)
        cv.waitKey(1)
        time.sleep(2)  # Wait for 2 seconds before capturing the next frame

    return captured_faces

def update_model_and_embeddings(new_name, captured_faces):
    global X, Y, model
    new_embeddings = []
    for img in captured_faces:
        img = np.expand_dims(img, axis=0)
        embedding = facenet.embeddings(img)
        new_embeddings.append(embedding[0])
    
    new_X = np.array(new_embeddings)
    new_Y = np.array([new_name] * len(new_X))
    
    X = np.concatenate([X, new_X])
    Y = np.concatenate([Y, new_Y])
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X, encoder.transform(Y))
    
    # Save the updated model
    with open(r"svm_model_160x160.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save the updated embeddings
    np.savez_compressed(r"faces_embeddings_done_4classes.npz", X, Y)

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        break
    
    cnt = 0
    unique_cnt = 0
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    frame_faces = []
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160)) # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        
        if(max(model.predict_proba(ypred)[0]) > 0.40):
            final_name = encoder.inverse_transform(face_name)[0]
        else:
            final_name = 'Unknown'

        if final_name != 'Unknown' and final_name not in unique_faces:
            unique_faces.add(final_name)  # Track unique face names
            log_unique_face(final_name) 

        if final_name != 'Unknown' and final_name not in frame_faces:
            frame_faces.append(final_name)
            log_face(final_name)  # Log to CSV
            
        elif final_name == 'Unknown':
            frame_faces.append(final_name)
            log_face(final_name)  # Log to CSV 
            
            # cv.putText(frame, "Add name? (y/n):", (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
            # cv.imshow("Face Recognition:", frame)
            # cv.waitKey(1)
            # choice = input("Do you want to add the name for the unknown face? (y/n): ")
            # if choice.lower() == 'y':
            #     name = input("Enter the name of the unknown face: ")
            #     captured_faces = capture_new_face(name)
            #     update_model_and_embeddings(name, captured_faces)
            #     unique_faces.add(name)
            #     log_unique_face(name)
            #     log_face(name)
            # else:
            #     print("Face not added.")

        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
    
    # Display the count of unique faces in frame detected
    cv.putText(frame, str(len(frame_faces)) + ' Face In Frame', (200, 440), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 0, 255), 2, cv.LINE_AA)

    # Display the count of unique faces detected
    cv.putText(frame, str(len(set(unique_faces))) + ' Known Faces Has Been Detected', (50, 475), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 2, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv.destroyAllWindows()
