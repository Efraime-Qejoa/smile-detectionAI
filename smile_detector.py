import cv2

#face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #haar algorithm is basically a bunch of faces someone build
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab webcam feed
webcam = cv2.VideoCapture(0) #you can also add video files inside brackets

# Show the current frame
while True:

    #Read current frame from webcam
    successful_frame_read, frame = webcam.read()

# If there's an error, abort
    if not successful_frame_read:
        break

    # change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale) #called multi scale so that it can detect different face sizes 
   

    # Run smile detector within each of the faces
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4) # the x & y are coordinates and w + h are width + height. the 3 numbers are your RGB/BGR colors so you can use any

        #Slicing // Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h , x:x+w]  #this only works in numpy
       
         # change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #scalefactor = how much you want to blur the image minneighbor = there need to be a certain redundant rectangles for that to be a smile
        #this is ran inside the face
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)  

        for (x_, y_, w_, h_) in smiles:

             # Draw a rectangle around the smile
             # the x & y are coordinates and w + h are width + height. the 3 numbers are your RGB/BGR colors so you can use any
             cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(200, 255, 255))

    cv2.imshow('Smile Detector', frame)

    #Display until user presses any key
    cv2.waitKey(1)

#Cleanup
webcam.release()
cv2.destroyAllWindows()


print("Complete")