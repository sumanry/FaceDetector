import cv2 

# Load image detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Image list from current directory images
image_list = ["mark.jpg", "mark1.jpg", "elon.jpg"]
# Iterate through list of images, detect faces, show and close
for img in image_list:
    # Load single image
    cap = cv2.VideoCapture(img)
    # Read image
    ret, img = cap.read()
    # Convert image in gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detects faces of different sizes in the input image
    faces = detector.detectMultiScale(gray, 1.3, 5)
    # Draw rectangle around face
    for(x, y, w, h) in faces:
        # To draw a rectangle around face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
    # Show the image in window
    cv2.imshow('img', img)
    k = cv2.waitKey(2000)
    #Close the window
    cap.release()
    # De_allocate any associated memory usage
    cv2.destroyAllWindows()
