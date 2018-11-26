import sys, cv2, numpy, os, pickle, copy, sqlite3
from new_gesture_app import add_gest
from PyQt5.QtCore import QTimer, QUrl, QFileInfo, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from keras.models import load_model

#referenced webcam tutorial from Life2Coding: https://www.youtube.com/watch?v=MUpC6z32bCA

class aslApp(QDialog):
    current_tab = None
    hist = None
    image_sample = None
    hist_exist = False
    gest_img_url = None
    gest_vid_url = None
    vid_frm = None
    vid_open = False
    vid_orien = 0
    trans_state = False
    prev_text = ""

    def __init__(self):
        super(aslApp, self).__init__()
        loadUi('UI_ASL_tab2.ui',self)
        self.image = None
        self.tabWidget.setCurrentIndex(0)
        current_tab = 0
        if os.path.isfile('hist') :
            hist_exist = True
            with open("hist", "rb") as file:
                self.hist = pickle.load(file)
        
        #organize button usage by tabs
        #if current_tab == 0 : #set histogram tab open
        self.captureBtn.clicked.connect(self.capture_hist)
        self.upImgBtn.clicked.connect(self.upload_image)
        self.upVidBtn.clicked.connect(self.upload_video)
        self.newGestBtn.clicked.connect(self.new_gest)
        self.rotCCWBtn.clicked.connect(self.rot_vid_ccw)
        self.rotCWBtn.clicked.connect(self.rot_vid_cw)
        self.tabWidget.currentChanged.connect(self.change_webcam)
        self.stopRadBtn.setChecked(True)
        self.clearBtn.clicked.connect(self.clear_output)
        #self.stopRadBtn.clicked.connect(lambda:self.radbtn_state(self.stopRadBtn))
        #self.startRadBtn.clicked.connect(lambda:self.radbtn_state(self.startRadBtn))
        self.get_image_size()
        self.start_webcam()
        self.tabWidget.blockSignals(False)

    def change_webcam(self):
        if self.current_tab == 0:
            self.current_tab = 1
            self.start_webcam()
        else:
            self.current_tab = 0
            self.stop_webcam()

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def stop_webcam(self):
        self.timer.stop()

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image = cv2.flip(self.image,1)

        x, y, w, h = 300, 100, 300, 300
        self.image_sample = self.image[y:y+h, x:x+w]

        display_img = copy.deepcopy(self.image)

        self.display_image(display_img,1)
        self.display_thresh()
    
    def display_image(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4 :
                qformat = QImage.Format_RGBA8888
            else :
                qformat = QImage.Format_RGB888
        x, y, w, h = 300, 100, 300, 300
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage = outImage.rgbSwapped()
        if window==1 :
            self.videoDisplay.setPixmap(QPixmap.fromImage(outImage))
            self.videoDisplay.setScaledContents(True)

    def display_thresh(self):
        #read saved pickle file, if it exists
        thresh = numpy.zeros((300,300,3), numpy.uint8)
        if self.hist_exist : #histogram exists, calculate thresh
            thresh = self.calc_thresh()

        outImage = QImage(thresh,thresh.shape[1],thresh.shape[0],thresh.strides[0],QImage.Format_RGB888)
        outImage = outImage.rgbSwapped()
        self.threshDisplay.setPixmap(QPixmap.fromImage(outImage))
        self.threshDisplay.setScaledContents(True)
        
        if self.startRadBtn.isChecked():
            self.translate(thresh)
       # else:
           # print("S T O P")

   


    def capture_hist(self):
        #everytime you capture skin color, save to pickle bytestream file
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        image_samp = self.build_squares(self.image)
        hsvCrop = cv2.cvtColor(image_samp, cv2.COLOR_BGR2HSV)
        hist_to_save = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_to_save, hist_to_save, 0, 255, cv2.NORM_MINMAX)
        self.hist_exist = True
        with open("hist", "wb") as file:
            pickle.dump(hist_to_save, file)

        if os.path.isfile('hist') :
            with open("hist", "rb") as file:
                self.hist = pickle.load(file)
    
    def calc_thresh(self):
        hsv = cv2.cvtColor(self.image_sample, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        ret,trsh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        trsh = cv2.merge((trsh,trsh,trsh))
        #trsh = cv2.cvtColor(trsh, cv2.COLOR_BGR2GRAY)
        return trsh

    def build_squares(self,img):
        x, y, w, h = 420, 140, 10, 10
        d = 10
        imgCrop = None
        crop = None
        for i in range(10):
            for j in range(5):
                if numpy.any(imgCrop == None):
                    imgCrop = img[y:y+h, x:x+w]
                else:
                    imgCrop = numpy.hstack((imgCrop, img[y:y+h, x:x+w]))
                #print(imgCrop.shape)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
                x+=w+d
            if numpy.any(crop == None):
                crop = imgCrop
            else:
                crop = numpy.vstack((crop, imgCrop)) 
            imgCrop = None
            x = 420
            y+=h+d
        return crop

        

    def upload_image(self):
        gest_img_file, _ = QFileDialog.getOpenFileName(self, 'Open file', '.',"Image files (*.jpg *.png)")
        self.gest_img_url = QUrl.fromLocalFile(gest_img_file)
        gest_img = cv2.imread(self.gest_img_url.fileName(),1)
        self.imgFileTxt.setPlainText(self.gest_img_url.path())
        pix = QPixmap.fromImage(QImage(gest_img, gest_img.shape[1], gest_img.shape[0],gest_img.shape[1]*3,QImage.Format_RGB888).rgbSwapped())
        pix = pix.scaled(240, 426, Qt.KeepAspectRatio)
        self.gestImgDisplay.setPixmap(pix)

    def upload_video(self):
        gest_vid_file, _ = QFileDialog.getOpenFileName(self, 'Open file', '.',"Video files (*.mp4 *.avi *.mov)")
        self.gest_vid_url = QUrl.fromLocalFile(gest_vid_file)
        self.vidFileTxt.setPlainText(self.gest_vid_url.path())
        gest_vid = cv2.VideoCapture(self.gest_vid_url.fileName())
        ret,frame = gest_vid.read()

        self.vid_frm = frame
        pix = QPixmap.fromImage(QImage(frame, frame.shape[1], frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888).rgbSwapped())
        pix = pix.scaled(300, 300, Qt.KeepAspectRatio)
        self.gestVidDisplay.setPixmap(pix)
        self.vid_open = True

    def rot_vid_ccw(self):
        if self.vid_open:
            self.vid_frm = cv2.rotate(self.vid_frm,rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = self.vid_frm

            pix = QPixmap.fromImage(QImage(frame, frame.shape[1], frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888).rgbSwapped())
            pix = pix.scaled(300, 300, Qt.KeepAspectRatio)
            self.gestVidDisplay.setPixmap(pix)

            #determine the orientation relative to current video orientation
            if self.vid_orien == 0:
                self.vid_orien = 1
            elif self.vid_orien == 1:
                self.vid_orien = 2
            elif self.vid_orien == 2:
                self.vid_orien = 3
            elif self.vid_orien == 3:
                self.vid_orien = 0

    def rot_vid_cw(self):
        if self.vid_open:
            self.vid_frm = cv2.rotate(self.vid_frm,rotateCode=cv2.ROTATE_90_CLOCKWISE)
            frame = self.vid_frm

            pix = QPixmap.fromImage(QImage(frame, frame.shape[1], frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888).rgbSwapped())
            pix = pix.scaled(300, 300, Qt.KeepAspectRatio)
            self.gestVidDisplay.setPixmap(pix)

            #determine the orientation relative to current video orientation
            if self.vid_orien == 0:
                self.vid_orien = 3
            elif self.vid_orien == 1:
                self.vid_orien = 0
            elif self.vid_orien == 2:
                self.vid_orien = 1
            elif self.vid_orien == 3:
                self.vid_orien = 2

    def new_gest(self):
        new_gest_app = add_gest(self.gest_img_url, self.gest_vid_url,self.IDTextEdit.toPlainText(),self.EngTextEdit.toPlainText(),self.vid_orien)
        if new_gest_app.isValid():
            self.gestOutTxt.insertPlainText("Now creating gesture in the database!\n")
            new_gest_app.start()
        else:
            self.gestOutTxt.insertPlainText("Error: must fill in all fields.\n")

    image_x = 0
    image_y = 0    

    def get_image_size(self):
        img = cv2.imread('gestures/0/100.jpg', 0)
        #return img.shape
        self.image_x,self.image_y = img.shape

    model = load_model('cnn_model_keras2.h5')

    def keras_process_image(self, img):
        img = cv2.resize(img, (self.image_x, self.image_y))
        img = numpy.array(img, dtype=numpy.float32)                     ###
        img = numpy.reshape(img, (1, self.image_x, self.image_y, 1))
        return img

    def keras_predict(self, model, image):
        processed = self.keras_process_image(image)
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def get_pred_text_from_db(self, pred_class):
        conn = sqlite3.connect("gesture_db.db")
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
        #print("translating...")

        tresh = trsh.copy()
        thresh = cv2.cvtColor(tresh, cv2.COLOR_BGR2GRAY)
        '''thresh.copy()'''
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            #print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

                pred_probab, pred_class = self.keras_predict(self.model, thresh) #save_img)
                #print(pred_class, pred_probab)

                if pred_probab*100 > 80:
                    text = self.get_pred_text_from_db(pred_class)
                    #print(text)
                    #splitted_text = self.split_sentence(text, 2)
                    #str = ''.join(splitted_text)
                    if text != self.prev_text:
                        self.transOutTxt.insertPlainText(text)
                        self.transOutTxt.insertPlainText(" ")
                        self.prev_text = text
        
        #find a way to not print the letter if the same as the previous letter
    def clear_output(self):
        self.transOutTxt.clear()
  
if __name__=='__main__':
    app = QApplication(sys.argv)
    window = aslApp()
    window.setWindowTitle('American Sign Language to English Text Application')
    window.show()
    sys.exit(app.exec_())