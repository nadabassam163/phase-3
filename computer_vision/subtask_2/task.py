import cv2
class Detect_Shape:
    def __init__(self,video_path):
        self.video_path=video_path
        self.cap=cv2.VideoCapture(self.video_path) #read the video
        print(self.cap.isOpened()) #check if the video is opened successfully
        self.frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #get the width of the video
        self.frame_height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #get the height of the video
        self.fps=self.cap.get(cv2.CAP_PROP_FPS) #get the fps (frames per second) of the video
        self.delay=int(1000/self.fps) #calculate the delay between frames
      

    def process_video(self):
        while True: 
            ret,frame=self.cap.read() #read the video frame by frame
            if not ret: #if the video is not read properly
                break
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #convert each frame to HSV (hue,saturation,value) color space
            red_mask1=cv2.inRange(hsv,(0,30,30),(15,255,255)) #create a mask for red color in each frame
            red_mask2=cv2.inRange(hsv,(165,30,30),(180,255,255)) #create a mask for red color in each frame
            red_mask=cv2.bitwise_or(red_mask1,red_mask2) #combine the two red masks
            blue_mask=cv2.inRange(hsv,(90,50,50),(130,255,255)) #create a mask for blue color in each frame
            red_contours,_=cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours of red color in each frame
            blue_contours,_=cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours of blue color in each frame
            for contour in red_contours: #loop through each contour of red color
                area=cv2.contourArea(contour) #calculate the area of each contour
                if area<100: #if the area is less than 100 pixels
                    continue
                (x,y),radius=cv2.minEnclosingCircle(contour) #get the minimum enclosing circle of each contour
                if radius==0:
                    continue
                circle_area=3.14*radius*radius #calculate the area of the minimum enclosing circle
                area_ratio=area/circle_area #calculate the area ratio of each contour
                if area_ratio>0.7: #if the area ratio is greater than 0.7
                    cv2.drawContours(frame,[contour],-1,(0,255,0),2) #draw contours of red color on each frame
               

            for contour in blue_contours: #loop through each contour of blue color
                area=cv2.contourArea(contour) #calculate the area of each contour
                if area<100: #if the area is less than 100 pixels
                    continue
                perimeter=cv2.arcLength(contour,True) #calculate the perimeter of each contour
                if perimeter==0: #if the perimeter is zero
                    continue
                approx=cv2.approxPolyDP(contour,0.04*perimeter,True) #approximate the contour to a polygon
                if len(approx)==4: #if the polygon has 4 vertices 
                    x,y,w,h=cv2.boundingRect(contour) #get the bounding rectangle of the polygon
                    aspect_ratio=float(w)/h #calculate the aspect ratio of the polygon
                    if aspect_ratio>=0.8 and aspect_ratio<=1.2: #if the aspect ratio is between 0.8 and 1.2
                        cv2.drawContours(frame,[contour],-1,(0,255,0),2) #draw contours of blue color on each frame    
           
            cv2.imshow('Shape Detection',frame) #display each frame with contour
            if cv2.waitKey(self.delay) & 0xFF == ord('q'):
                break
            
        self.cap.release() #close the video file
        cv2.destroyAllWindows() #close all OpenCV windows


video_path=r'C:\\Users\\nadab\\OneDrive\\Desktop\\phase-3\\computer_vision\\thrown_shapes_noisy_30s.mp4' #path to the video file
shape_detector=Detect_Shape(video_path) #create an object of the Detect_Shape class
shape_detector.process_video() #call the process_video method
