import cv2
import face_recognition
from glob import glob

known_encodings = []
known_names = []

for f in glob("database/*/*.jpg"):
    face_name = f.split("\\")[1]
    print(face_name)
    image = cv2.imread(f) #BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB image conversion

    face_locations = face_recognition.face_locations(image_rgb, model="cnn")
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
    for encoding in face_encodings:
        known_encodings.append(encoding)
        known_names.append(face_name)
data = {"encodings": known_encodings, "names": known_names}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
video = cv2.VideoCapture("amy-sheldon.mp4")

while (video.isOpened()):
    ret, frame = video.read()

    if ret == True:
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting frame to RGB
        video_encodings = face_recognition.face_encodings(rgb_frame)
        names = []

        for encoding in video_encodings:

            matches = face_recognition.compare_faces(data["encodings"],encoding)
            name = "Unknown"

            if True in matches:
                matched_positions = [i for (i,b) in enumerate(matches) if b]
                counts = {}

                for i in matched_positions:
                    name = data["names"][i]
                    counts[name] = counts.get(name,0) + 1
                    name = max(counts, key=counts.get)

                names.append(name)

                for ((x, y, w, h), name) in zip(faces, names):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video.release()
cv2.destroyAllWindows()
