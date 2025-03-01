import os
import sys
import cv2
import numpy as np
import pandas as pd
import create_csv  


if len(sys.argv) < 2:
    print("Error: Please provide the test image path.")
    sys.exit()

test_img = sys.argv[1]


cv2_base_dir = os.path.dirname(cv2.__file__)
CASCADE_PATH = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')

if not os.path.exists(CASCADE_PATH):
    print("Error: Haarcascade file not found!")
    sys.exit()

faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

def train():
    """Trains face recognizer using LBPH and maps names to labels."""
    create_csv.create()  

    if not os.path.exists("train_faces.csv") or os.stat("train_faces.csv").st_size == 0:
        print("Error: train_faces.csv is empty or missing!")
        sys.exit()

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    data = pd.read_csv("train_faces.csv").values
    images = []
    labels = []
    label_to_name = {}
    name_to_label = {}
    for img_path, name in data:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if name not in name_to_label:
            label = len(name_to_label)
            name_to_label[name] = label
            label_to_name[label] = name

        images.append(gray)
        labels.append(name_to_label[name])  

    if len(images) == 0:
        print("Error: No valid images found for training!")
        sys.exit()

    face_recognizer.train(images, np.array(labels))
    return face_recognizer, label_to_name

def test(test_img, face_recognizer, label_to_name):
    """Tests a given image and displays the person's name."""
    if not os.path.exists(test_img):
        print(f"Error: Test image '{test_img}' not found!")
        sys.exit()

    image = cv2.imread(test_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    for (x, y, w, h) in faces:
        sub_img = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
        pred_label, confidence = face_recognizer.predict(sub_img)

       
        person_name = label_to_name.get(pred_label, "Unknown")

       
        cv2.putText(image, f"{person_name}", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")


if __name__ == "__main__":
    face_recog, label_to_name = train()
    test(test_img, face_recog, label_to_name)
