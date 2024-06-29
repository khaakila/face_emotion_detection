import cv2
from detect_faces import detect_faces
from classify_emotion import classify_emotion

def main():
    
    image_path = r'C:\Users\USER\Pictures\Camera Roll\WIN_20240606_19_30_00_Pro.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {C:\Users\USER\Pictures\Camera Roll\WIN_20240606_19_30_00_Pro.jpg}")
        return

    faces = detect_faces(image)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        emotion = classify_emotion(face)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
