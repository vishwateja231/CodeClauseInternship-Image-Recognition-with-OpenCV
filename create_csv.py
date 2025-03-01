import os
import csv
import cv2

DATASET_PATH = "Training_Data"
OUTPUT_PATH = "Training_Faces"
CSV_FILE = "train_faces.csv"

cv2_base_dir = os.path.dirname(cv2.__file__)
CASCADE_PATH = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise FileNotFoundError("Error: Haarcascade file not found!")

def create():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    try:
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "label"])

            for person in os.listdir(DATASET_PATH):
                person_path = os.path.join(DATASET_PATH, person)

                if os.path.isdir(person_path):
                    for image in os.listdir(person_path):
                        image_path = os.path.join(person_path, image)
                        file_name, file_ext = os.path.splitext(image.lower())

                        if file_ext in [".jpg", ".jpeg", ".png"]:
                            img = cv2.imread(image_path)
                            if img is None:
                                print(f"Warning: Unable to read image {image_path}")
                                continue

                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                            for i, (x, y, w, h) in enumerate(faces):
                                face = gray[y:y + h, x:x + w]
                                face_filename = f"{person}_{i}.jpg"
                                face_path = os.path.join(OUTPUT_PATH, face_filename)

                                cv2.imwrite(face_path, face)
                                writer.writerow([face_path, person])

        print(f"✅ CSV file '{CSV_FILE}' created successfully!")
    except IOError as e:
        print(f"Error: Unable to write to CSV file {CSV_FILE}. {e}")

if __name__ == "__main__":
    create()
