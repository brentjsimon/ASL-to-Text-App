import cv2
from PyQt5.QtCore import QUrl
import numpy, pickle, os, sqlite3, random

class add_gest(object):
    img_url = None
    vid_url = None
    id_num = None
    eng_trans = None
    vid_orien = 0
    valid = True
    hist = None
    num_pics = 0

    def __init__(self,gest_img_url,gest_vid_url,id_text,eng_text,orien):
        print("Button Pressed")
        all_vars_here = True
        
        if gest_img_url != None:
            self.img_url = gest_img_url
            print(gest_img_url.path())
        else:
            all_vars_here = False

        if gest_vid_url != None:
            self.vid_url = gest_vid_url
            print(gest_vid_url.path())
        else:
            all_vars_here = False

        if id_text != "":
            self.id_num = id_text
            print(id_text)
        else:
            all_vars_here = False

        if eng_text != "":
            self.eng_trans = eng_text
            print(eng_text)
        else:
            all_vars_here = False

        self.vid_orien = orien

        self.valid = all_vars_here

    def start(self):
        print("Starting function!")
        
        #step 1. create histogram from img
        self.create_hist()

        #step 2. process video frame by frame
        self.vid_to_data()

        #step 3. flip images
        self.flip()

        #gesture data added to database
        #return True if no error occured
        
    def create_hist(self):
        #get image data from the url
        img = cv2.imread(self.img_url.fileName(),1)
        
        #divide image up into squares to sample from
        #push these squares into a numpy array
        img_crop = self.crop_samples(img)
        hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv_crop],[0,1],None,[180, 256], [0, 180, 0, 256])
        #also try the histogram tutorial, use [256], [0,256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

    def vid_to_data(self):        
        #create while loop
        #iterate through video frame by frame (use read() fn) until
        #another frame exists or reached max number of frames needed

        #calculate the current frame of the video and display it
        vid = cv2.VideoCapture(self.vid_url.fileName())
        pic_num = 0

        while True:
            ret,frm = vid.read()
            
            #change accordingly? to get approx 600-900 frames
            #current video is up to 30 fps for 30 sec each
            if pic_num >= 900 or not ret:
                break;

            #MAY NEED TO CHANGE ROTATION DEPENDING ON VIDEO
            if self.vid_orien == 1:
                frm = cv2.rotate(frm,rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.vid_orien == 2:
                frm = cv2.rotate(frm,rotateCode=cv2.ROTATE_180)
            elif self.vid_orien == 3:
                frm = cv2.rotate(frm,rotateCode=cv2.ROTATE_90_CLOCKWISE)
            
            hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)

            dst = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            
            if not os.path.exists("gestures/"+str(self.id_num)):
                os.mkdir("gestures/"+str(self.id_num))

            #thresh = thresh[y:y+h, x:x+w]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            image_x, image_y = 50, 50
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    pic_num += 1
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                    save_img = cv2.resize(thresh, (image_x, image_y))
                    rand = random.randint(0, 10)
                    if rand % 2 == 0:
                        save_img = cv2.flip(save_img, 1)
                        
                    cv2.imwrite("gestures/"+str(self.id_num)+"/"+str(pic_num)+".jpg", save_img)
                    #cv2.imshow("Thresh", thresh)
        
        self.num_pics = pic_num
    
    #NEED TO FIX THIS
    def flip(self):
        for n in os.listdir("gestures"):
            for i in range(self.num_pics):
                img = cv2.imread("gestures/"+n+"/"+str(i+1)+".jpg", 0)
                img = cv2.flip(img, 1)
                cv2.imwrite("gestures/"+n+"/"+str(i+1+self.num_pics)+".jpg", img)

    def crop_samples(self, img):
        height, width = img.shape[:2]
        mult = 5
        y_div = height // (16 * mult)
        x_div = width // (9 * mult)
        crop = None
        img_crop = None

        #create 100 samples
        #from pixel y = 0 to height (16*mult divisions)
        for y in range(0, height, y_div):
            #from pixel x = 0 to width (9*mult divisions)
            for x in range(0, width, x_div):
                if numpy.any(img_crop == None):
                    img_crop = img[y:y+y_div, x:x+x_div]
                else:
                    img_crop = numpy.hstack((img_crop, img[y:y+y_div, x:x+x_div]))
            if numpy.any(crop == None):
                crop = img_crop
            else:
                crop = numpy.vstack((crop, img_crop))
            img_crop = None
        
        return crop


    def isValid(self):
        return self.valid