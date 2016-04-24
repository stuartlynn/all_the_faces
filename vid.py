import numpy as np
import cv2
from IPython import embed
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_lfw_people

cap = cv2.VideoCapture(0)
min_size = (20, 20)
image_scale = 2
haar_scale = 2.0
min_neighbors = 5
haar_flags = 0

#train
n_components = 150
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, eh, ew = lfw_people.images.shape

aspect = float(eh)/float(ew)

print 'eface width : {width} height: {height} aspect: '.format( width=ew, height=eh, aspect=aspect)
x= lfw_people.data
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(x)
eigenfaces = pca.components_.reshape((n_components, eh, ew))


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('/Users/stuartlynn/Documents/Processing/libraries/opencv_processing/library/cascade-files/haarcascade_frontalface_default.xml')
    # # cv2.CvtColor(frame, gray, cv2.CV_BGR2GRAY)
    # gray = cv.CreateImage((frame.width,img.height), 8, 1)
    # # scale input image for faster processing
    print 'running cascade'
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30) ,
        flags =  cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print 'faces %',faces
    
    if len(faces)>=1:
        biggest_face = np.argmax([ face[2]*face[3] for face in faces],axis=0)
        
        print "Found {0} faces! biggest is {1}".format(len(faces),biggest_face)

        (x, y, w, h) = faces[0]
        w2 = 50
        h2 = 37


        new_h = int(w*37.0/50.0)
            # the input to cv2.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
            # pt1 = (int(x * image_scale), int(y * image_scale))
            # pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
        cv2.rectangle(frame, (x,y),(x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h,x:x+h]

        scaledFace = cv2.resize(face,(ew,eh))

        lin_face  = scaledFace.reshape(ew*eh)
    
        c = pca.transform(lin_face)

        comb = np.average( eigenfaces, weights=c[0], axis=0)
        print np.shape(comb)
        cv2.imshow('face',face)
        # cv2.imshow('face reconstructed',cv2.resize(np.multiply(comb,1000), (0,0), fx=10, fy=10))

        # cv2.imshow('scaled face ', cv2.resize(np.multiply( comb,1000),(0,0), fx=10, fy=10))
        cv2.imshow('eface', cv2.resize(np.multiply(eigenfaces[50],8000), (0,0), fx=10, fy=10))

    # # cv.ShowImage("result", img)
    # # Display the resulting frame
    cv2.imshow('faces_found',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
