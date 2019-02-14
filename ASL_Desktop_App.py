##### NOTES #####

# basic structure of application based off of https://github.com/evilport2/sign-language
# main aspects directly taken from evilport include
#  - opencv skin color detection
#  - basic translation algorithm
# aspects that have been revised/added for this project include
#  - use of cnn tensorflow instead of keras
#  - dynamic hand detection
#  - use of original pyqt5 gui (instead of opencv)

##### ADDITIONAL REFERENCES #####

# [1] Life2Coding https://www.youtube.com/watch?v=MUpC6z32bCA
# [2] Real Python https://realpython.com/face-recognition-with-python/
# [3] pyimagesearch https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/


##### IMPORTS #####

import sys, cv2, numpy, os, pickle, copy, sqlite3, imutils
import tensorflow as tf
from cnn_tf import cnn_model_fn

from PyQt5.QtCore import QTimer, QUrl, QFileInfo, Qt
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi


##### APP CLASS #####

class aslApp(QDialog):
    ##### GLOBAL VARIABLES #####

    hist = None
    image = None
    hist_exist = False
    prev_text = ""
    frame_count = 0
    img_x, img_y = 0, 0


    ##### INITIALIZATION #####

    def __init__(self):
        super(aslApp, self).__init__()

        #load pyqt ui file
        loadUi('ASL_Desktop.ui',self)

        if os.path.isfile('data/hist') :
            self.hist_exist = True
            with open("data/hist", "rb") as file:
                self.hist = pickle.load(file)
        
        casc_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(casc_path)

        #initialize buttons
        self.captureBtn.clicked.connect(self.capture_hist)
        self.clrHistBtn.clicked.connect(self.clear_hist)
        self.clearBtn.clicked.connect(self.clear_output)

        self.get_image_size()
        self.start_webcam()


    ##### GUI DISPLAY FUNCTIONS #####

    #referenced webcam tutorial from Life2Coding [1]
    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    #reads in new frame to be processed
    def update_frame(self):
        self.frame_count += 1

        __,self.image=self.capture.read()
        self.image = cv2.flip(self.image,1)

        display_img = copy.deepcopy(self.image)
        self.display_image(display_img,1)

        self.display_thresh()


    #displays the unprocessed video frame from the live webcam feed
    def display_image(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4 :
                qformat = QImage.Format_RGBA8888
            else :
                qformat = QImage.Format_RGB888
       
        #show histogram sampling boxes
        if self.histCheckBox.isChecked():
            x, y, w, h, d = 350, 100, 15, 15, 15
            for i in range(10):
                for j in range(5):
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
                    x += w+d
                x = 350
                y += h+d
        
        #show static translation box
        elif not self.handCheckBox.isChecked():
            x, y, w, h = 300, 100, 300, 300
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        out_img = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        out_img = out_img.rgbSwapped()

        if window == 1:
            self.videoDisplay.setPixmap(QPixmap.fromImage(out_img))
            self.videoDisplay.setScaledContents(True)


    def display_thresh(self):
        #initialize output images (numpy array rep)
        cam_trsh = numpy.zeros((640,480,3), numpy.uint8)
        out_trsh1 = numpy.zeros((300,300,3), numpy.uint8)
        out_trsh2 = numpy.zeros((300,300,3), numpy.uint8)
        out_trsh2[::] = 255 #default white
        
        #calculate thresh, if histogram exists
        if self.hist_exist:
            #will detect hand anywhere in the frame
            if self.handCheckBox.isChecked():
                #referenced face detection tutorial from Real Python [2]
                # Detect faces in the image
                faces = self.face_cascade.detectMultiScale(
                    cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1, #bring up to 1.3 for faster but less accurate detection
                    minNeighbors=5,
                    minSize=(60, 60),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                #cover faces in the image
                for (x, y, w, h) in faces:
                    cv2.rectangle(self.image, (x-25, y-25), (x+w+25, y+h+25), (0, 0, 0), -1)
            
                #calculate thresh of the whole frame
                cam_trsh = self.calc_thresh(self.image)

                #referenced contour detection tutorial from pyimagesearch [3]
                gray = cv2.cvtColor(cam_trsh.copy(), cv2.COLOR_BGR2GRAY)
                cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                if len(cnts) > 0:
                    #calculate boundaries of the largest contour
                    c = max(cnts, key=cv2.contourArea)                    
                    hull = cv2.convexHull(c)
                    x,y,w,h = cv2.boundingRect(c)
                    
                    #calculate the center
                    cntr_x, cntr_y = x+w//2, y+h//2

                    #calculate two points for the square cropping bounds
                    if w > h :
                        l = w // 2
                        x1, y1, x2, y2 = cntr_x-l, y+h-l*2, cntr_x+l, y+h                        
                    else:
                        l = h // 2
                        x1, y1, x2, y2 = cntr_x-l, cntr_y-l, cntr_x+l, cntr_y+l
                    
                    #output thresh for translation
                    out_trsh1 = cam_trsh[y1:y2, x1:x2].copy()

                    #output thresh with contours
                    cv2.drawContours(cam_trsh, [c], -1, (255, 0, 255), 2) #magenta contour
                    cv2.drawContours(cam_trsh, [hull], -1, (0, 255,255), 2) #yellow convex hull              
                    out_trsh2 = cam_trsh[y1:y2, x1:x2].copy()

                    #bounding rectangle drawn on webcam tresh
                    cv2.rectangle(cam_trsh,(x1,y1),(x2,y2),(255,255, 0), 2) #cyan square

            else: #will only traslate hand detected within a static box                                
                x, y, w, h = 300, 100, 300, 300
                out_trsh1 = self.calc_thresh(self.image[y:y+h, x:x+w])
                cam_trsh = self.calc_thresh(self.image)

        #output webcam processed thresh
        out_img = QImage(cam_trsh,cam_trsh.shape[1],cam_trsh.shape[0],cam_trsh.strides[0],QImage.Format_RGB888)
        self.threshDisplay.setPixmap(QPixmap.fromImage(out_img.rgbSwapped()))
        self.threshDisplay.setScaledContents(True)

        #output isolated thresh for translation
        out_img = QImage(out_trsh1,out_trsh1.shape[1],out_trsh1.shape[0],out_trsh1.strides[0],QImage.Format_RGB888)
        self.threshDisplay1.setPixmap(QPixmap.fromImage(out_img.rgbSwapped()))
        self.threshDisplay1.setScaledContents(True)

        #output thresh with contours
        out_img = QImage(out_trsh2,out_trsh2.shape[1],out_trsh2.shape[0],out_trsh2.strides[0],QImage.Format_RGB888)
        self.threshDisplay2.setPixmap(QPixmap.fromImage(out_img.rgbSwapped()))
        self.threshDisplay2.setScaledContents(True)        

        #will translate the output thresh when translate box is checked
        #translates every ten frames to slow down the tranlsation rate
        if self.transCheckBox.isChecked() and self.frame_count > 10:
            self.translate(out_trsh1) 
            self.frame_count = 0


    ##### TRANSLATION FUNCTIONS #####

    classifier = tf.estimator.Estimator(model_dir="data/cnn_model5", model_fn=cnn_model_fn)


    def tf_process_image(self, img):
        img = cv2.resize(img, (self.img_x, self.img_y))
        img = numpy.array(img, dtype=numpy.float32)
        numpy_array = numpy.array(img)
        return numpy_array


    def tf_predict(self, classifier, image):
        global prediction
        processed_array = self.tf_process_image(image)
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":processed_array}, shuffle=False)
        pred = classifier.predict(input_fn=pred_input_fn)
        prediction = next(pred)
        return prediction


    def get_pred_text_from_db(self, pred_class):
        conn = sqlite3.connect("data/newgesture_db.db")
        cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
        cursor = conn.execute(cmd)
        for row in cursor:
            return row[0]


    def split_sentence(self, text, num_of_words):
        '''
        Splits a text into group of num_of_words
        '''
        list_words = text.split(" ")
        length = len(list_words)
        splitted_sentence = []
        b_index = 0
        e_index = num_of_words
        while length > 0:
            part = ""
            for word in list_words[b_index:e_index]:
                part = part + " " + word
            splitted_sentence.append(part)
            b_index += num_of_words
            e_index += num_of_words
            length -= num_of_words
        return splitted_sentence


    def translate(self, trsh):
        tresh = trsh.copy()
        thresh = cv2.cvtColor(tresh, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

                prediction = self.tf_predict(self.classifier, thresh) 
                pred_class = prediction.get("classes")
                pred_prob  = prediction.get("probabilities")

                if max(pred_prob)*100 > 98:
                    text = self.get_pred_text_from_db(pred_class)
                    if text != self.prev_text:
                        self.transOutTxt.insertPlainText(text)
                        self.transOutTxt.insertPlainText(" ")
                        self.prev_text = text
 

    ##### BUTTONS #####

    #everytime you capture skin color, save histogram to pickle bytestream file
    def capture_hist(self):
        #opencv image processing to calculate the histogram
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        image_samp = self.crop_samples(self.image)
        hsvCrop = cv2.cvtColor(image_samp, cv2.COLOR_BGR2HSV)
        hist_to_save = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_to_save, hist_to_save, 0, 255, cv2.NORM_MINMAX)

        self.hist_exist = True
        self.hist = hist_to_save

        with open("data/hist", "wb") as file:
            pickle.dump(hist_to_save, file)


    #delete the histogram & remove from the app
    def clear_hist(self):
        self.hist = None
        self.hist_exist = False

        if os.path.exists("data/hist"):
            os.remove("data/hist")


    #clear the output translation textbox
    def clear_output(self):
        self.transOutTxt.clear()


    ##### HELPER FUNCTIONS #####

    def crop_samples(self,img):
        x, y, w, h, d = 350, 100, 15, 15, 15
        img_crop = None
        crop = None
        for i in range(10):
            for j in range(5):
                if numpy.any(img_crop == None):
                    img_crop = img[y:y+h, x:x+w]
                else:
                    img_crop = numpy.hstack((img_crop, img[y:y+h, x:x+w]))
                x+=w+d
            if numpy.any(crop == None):
                crop = img_crop
            else:
                crop = numpy.vstack((crop, img_crop)) 
            img_crop = None
            x = 350
            y+=h+d
        return crop


    def calc_thresh(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        __,trsh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        trsh = cv2.merge((trsh,trsh,trsh))
        return trsh


    def get_image_size(self):
        img = cv2.imread('data/sample_image.jpg', 0)
        self.img_x, self.img_y = img.shape


##### MAIN FUNCTION #####

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = aslApp()
    window.setWindowTitle('American Sign Language to English Text Application')
    window.show()
    sys.exit(app.exec_())