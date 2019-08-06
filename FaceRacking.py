# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:18:47 2019

@author: Umar Sheikh (Spy)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2
import PIL
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import smtplib,ssl
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tkinter as tk
import tkinter.filedialog 
global class_id
import threading




global facecount
facecount=0
def faceDetect(frame,img):
    global facecount
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    margin=30
    cascPath = 'D:/Programs/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(frame)
    faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    print("No Of Face Detected in Frame %d: "%img,len(faces))
    text.delete(1.0,tk.END)
    text.insert(tk.INSERT,"Faces Extracted from Frame %d "%img)
    #text.insert(tk.INSERT," "+str(len(faces)))
    
    text.insert(tk.INSERT,"\n")
    for (x, y, width, height) in faces:
        #cv2.rectangle(frame, (x, y), (x+width+margin, y+height+margin), (0, 255, 0), 2)
        faceimg = frame[y-margin:y+height+margin, x-margin:x+width+margin]
        try:
            faceimg = cv2.resize(faceimg, (500, 500))
            cv2.imwrite("Faces/Image-%07d.jpg"% facecount , faceimg)
            print("Faces/Image-%07d.jpg"%facecount)
            facecount += 1
        except Exception as e:
            print(str(e))

def get_contacts(filename):
    
    names=[]
    emails=[]
    with open(filename,mode='r',encoding='utf-8') as contact_file:
        for contact in contact_file:
            names.append(contact.split()[0])
            emails.append(contact.split()[1])
    return names,emails

def get_template(filename):
    
    with open(filename,mode='r',encoding='utf-8') as template_file:
        template=template_file.read()
    return Template(template)


input_video="Data_Set/00000.MTS"
input_video2="Data_Set/video1.mp4"
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
update="train_img/"

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize =50  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture1=[]
        video_capture1.append(cv2.VideoCapture(input_video))
        video_capture1.append(cv2.VideoCapture(input_video2))
        
        #video_capture1.append(cv2.VideoCapture("http://192.168.0.100:8080/video"))
        
        #video_capture1.append(cv2.VideoCapture("http://192.168.43.1:8080/video"))
        c = 0


from twilio.rest import Client

def StartRecognition():
    input_video="Data_Set/00000.MTS"
    input_video2="Data_Set/video1.mp4"
    modeldir = './model/20170511-185253.pb'
    classifier_filename = './class/classifier.pkl'
    npy='./npy'
    train_img="./train_img"
    update="train_img/"

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
    
            minsize =60  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
    
            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
    
    
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
    
            video_capture1=[]
            video_capture1.append(cv2.VideoCapture(input_video))
            video_capture1.append(cv2.VideoCapture(input_video2))
            
            #video_capture1.append(cv2.VideoCapture("http://192.168.0.100:8080/video"))
            
            #video_capture1.append(cv2.VideoCapture("http://192.168.43.1:8080/video"))
            c = 0
    
    
            print('Start Recognition')
            presvTime = 0
            timeTaken=0
            totalFrame=1
            Attend=dict()
            for i in HumanNames:
                Attend[i]=0
            while True:
                if StopRec==True:
                    break
                MainFrame = []
                n=1
                for video_capture in video_capture1:
                    t=0
                    start1=time.time()
                    while(t<=int(timeTaken*30)):
                        ret, fram = video_capture.read()
                        t+=1
                    end1=time.time()
                    print("Skipping Time:",end1-start1)
                    #if(ret):
                        #fram = cv2.resize(fram, (0,0), fx=0.5, fy=0.5)
                    #else:
                    #    print("Camera%d Not Detected"%n)
                    #    continue
                    MainFrame.append(fram)
                    #cv2.imshow('video%d'%i, fram)
                    n+=1
                start=time.time()
                if (len(MainFrame)==0):
                    print("No Camera Detected Terminating Program")
                    break
                curTime = time.time()+1    # calc fps
                timeF = frame_interval
                reIdentify=[]
                NoOfProb=[]
                print()
                n=1
                for frame in MainFrame:
                    if n==3:
                        n=1
                    print()
                    print("=============Camera %d============="%n)
                    n+=1
                    if (c % timeF == 0):
                        find_results = []
    
                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)
    
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]
    
                            cropped = None
                            scaled = None
                            scaled_reshape = None
                            bb = np.zeros((nrof_faces,4), dtype=np.int32)
                            NoOfFaces=[]
                            for i in range(nrof_faces):
                                emb_array = np.zeros((1, embedding_size))
    
                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]
    
                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                    print('Face is very close!')
                                    continue
                                margin = 50
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                im = frame[bb[i][1]-margin:bb[i][3]+margin, bb[i][0]-margin:bb[i][2]+margin, :]
                                cropped = facenet.flip(cropped, False)
                                scaled  = np.array(PIL.Image.fromarray(cropped).resize((image_size, image_size),PIL.Image.BILINEAR))
                                scaled  = cv2.resize(scaled, (input_image_size,input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled  = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1,input_image_size,input_image_size,3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    
                                NoOfFaces.append(best_class_indices)
                                NoOfProb.append(best_class_probabilities)
                                if(best_class_probabilities>=0.80 and up_var.get()==1):
                                    try:
                                        
                                        im = cv2.resize(im, (500, 500))
                                      #  cv2.imshow("img"+str(i),im)
                                        update="test/"
                                        #print(best_class_indices)
                                        update+=HumanNames[int(best_class_indices)]
                                        
                                        #print(update)
                                        if not os.path.exists(update):
                                            os.makedirs(update)
                                        list_of_files = os.listdir(update)
                                        full_path = [update+"/{0}".format(x) for x in list_of_files]
                    
                                        if len([name for name in list_of_files]) >= 50:
                                            oldest_file = min(full_path, key=os.path.getctime)
                                            print(oldest_file)
                                            update=oldest_file
                                            os.remove(oldest_file)
                                        #update+="/image.png"
                                        #im = cv2.resize(im,(182,182))
                                        cv2.imwrite(update, im)
         
                                    except Exception as e:
                                            print(str(e))
                                    
                            
                            reIdentify.append(NoOfFaces)        
                
                if(len(reIdentify)>1):
                    print("===========ReIdentification==============")
                    for i in range(len(reIdentify[0])):
                        if reIdentify[0][i] in reIdentify[1]:
                            #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (192, 192, 192), 2)
                            #plot result idx under box
                            #text_x = bb[i][0]
                            #text_y = bb[i][3] + 20
                            #print("Prob: ",NoOfProb[i])
                            #print('Re-Identified Result: ', reIdentify[0][i])
                            
                            update="train_img/"
                            update+=HumanNames[int(reIdentify[0][i])]
                            print(update)
                            #faceDetect('Camera2',frame)
                            #cv2.imwrite('Path/Image.jpg', frame)
                            Attend[HumanNames[int(reIdentify[0][i])]]+=1
                            if((int(reIdentify[0][i])==7) | (int(reIdentify[0][i])==12)):
                                print('Present: '+ HumanNames[int(reIdentify[0][i])]+'\t %.2f'%round((Attend[HumanNames[int(reIdentify[0][i])]]/totalFrame)*100,2)+'%')
                            elif(int(reIdentify[0][i])==4):
                                print('Present: '+ HumanNames[int(reIdentify[0][i])]+'\t\t\t %.2f'%round((Attend[HumanNames[int(reIdentify[0][i])]]/totalFrame)*100,2)+'%')
                            else:
                                print('Present: '+ HumanNames[int(reIdentify[0][i])]+'\t\t %.2f'%round((Attend[HumanNames[int(reIdentify[0][i])]]/totalFrame)*100,2)+'%')
                            result_names = HumanNames[int(reIdentify[0][i])]
                            #cv2.putText(frame, result_names, (text_x, text_y),cv2.FONT_HERSHEY_PLAIN,
                            #            0.7, (0,0,255), thickness=1, lineType=3)
                            #cv2.imwrite("Predictions/img-%d.jpg"%totalFrame,frame)
                            
                
                        else:
                            if NoOfProb[i]>=0.82:
                                Attend[HumanNames[int(reIdentify[0][i])]]+=1
                                print("Prob: ",NoOfProb[i])
                                print('Present: '+ HumanNames[int(reIdentify[0][i])]+'\t %.2f'%round((Attend[HumanNames[int(reIdentify[0][i])]]/totalFrame)*100,2)+'%')
                                #text.insert(tk.INSERT,HumanNames[int(reIdentify[0][i])])
                                #text.insert(tk.INSERT,"\n")
                                #text.pack()
                            #else:
                            #    print("Not Identified: " ,HumanNames[int(reIdentify[0][i])])
                text.delete(1.0,tk.END)
                for i in HumanNames:
                    Attendence = (Attend[i]/totalFrame)*100
                    if(Attendence>=33.333):
                        text.insert(tk.INSERT,"Present: \t")
                        text.insert(tk.INSERT,i)
                        text.insert(tk.INSERT,"\n")
                        #text.pack()
                    else:
                        text.tag_config('warning',foreground="red")
                        text.insert(tk.INSERT,"Absent: \t","warning")
                        text.insert(tk.INSERT,i,"warning")
                        text.insert(tk.INSERT,"\n")
                        #text.pack()
                        
                        
                end=time.time()
                print("Time Taken: ",end-start)
                timeTaken=end-start
                totalFrame+=1
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            video_capture.release()
            cv2.destroyAllWindows()
    if(email_var.get()==1):
        print("Sending Email")
        report=""
        for i in HumanNames:
            Attendence = (Attend[i]/totalFrame)*100
            if(Attendence>=33.333):
                #print('Present with %.2f'%round(Attendence,2)+'% \t:', i)
                report =report +'<p style="margin: 0;padding: 0;"><b>Present</b> with %.2f'%round(Attendence,2)+'%\t: '+i+'</p>'
            else:
                #print('Absent  with %.2f'%round(Attendence,2)+'% \t:', i)
                report =report +'<p style="color: red;margin: 0;padding: 0;"><b>Absent</b> with %.2f'%round(Attendence,2)+'%\t: '+i+'</p>'
        #print(report)
        
        names,emails    = get_contacts('contact.txt')
        message_template= get_template('message.txt')
        subject='Attendence Report: IML'
        course='<b>Introduction To Machine Learning</b>'
        Signature = '<br>Team FaceRacking<br>University Of Central Punjab'
        #s= smtplib.SMTP(host='smtp.gmail.com',port=587)
        #s.starttls()
        #s.login('faceracking@gmail.com','teamfaceracking')
        sender='faceracking@gmail.com'
        password='teamfaceracking'
        for name, email in zip(names,emails):
            message= MIMEMultipart()
            messageDetails=message_template.substitute(PERSON_NAME=name,COURSE=course,REPORT_DETAIL=report,SIGNATURE=Signature)
            #print(messageDetails)
            message['From'] = sender
            message['To'] = email
            message['Subject']=subject
            message.attach(MIMEText(messageDetails,'html'))
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender, password)
                server.sendmail(sender, email, message.as_string())
                print("Done")
            #s.send_message(message)
            del message
            #s.quit()
    if(sms_var.get()==1):
        print("Sending SMS")
        client = Client("AC46c4b758dc29b405a46dc64503b86c19", "343b90764bf6d47e559f2fe6e15baa6b")
        # change the "from_" number to your Twilio number and the "to" number
        # to the phone number you signed up for Twilio with, or upgrade your
        # account to send SMS to any phone number
        message = "Dear Parent,\nBelow is the Attendence of your Son/daughter for the course 'Introduction To Machine Learning'\nPresent	: Abdur rehman L1F15BSCS0430\n\n\n\n\n\nTeam FaceRacking\nUniversity Of Central Punjab"
        client.messages.create(to="+923204690469", 
                               from_="+18036102865", 
                               body=message)
        print("Done")






global text
global stopRec
global x
from classifier import training
from preprocess import preprocesses






def StartPreprocess():
    global StopPre
    preprocess.configure(state=tk.DISABLED)
    input_datadir = './train_img'
    output_datadir = './pre_img'
    obj=preprocesses(input_datadir,output_datadir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    StopPre=True
    preprocess.configure(state=tk.ACTIVE)
    text.insert(tk.INSERT,"\nCompleted...")
    timer.destroy()
    

def StartTraining():
    global StopTrain
    train.configure(state=tk.DISABLED)
    datadir = './pre_img'
    modeldir = './model/20170511-185253.pb'
    classifier_filename = './class/classifier.pkl'
    print ("Training Start")
    obj=training(datadir,modeldir,classifier_filename)
    get_file=obj.main_train()
    print('Saved classifier model to file "%s"' % get_file)
    StopTrain=True
    train.configure(state=tk.ACTIVE)
    text.insert(tk.INSERT,"\nCompleted...")
    timer.destroy()
    
from tkinter import messagebox
global StopTrain
StopTrain=True
def train():
    if(StopRec==False):
        messagebox.showerror("Error", "Cannot Start Training While Classroom Attendence is Running")
    elif(StopFram == False):
        messagebox.showerror("Error", "Cannot Start Training While Frames Are Extracting")
    elif(StopFace == False):
        messagebox.showerror("Error", "Cannot Start Training While Faces Are Extracting")
    elif(StopPre == False):
        messagebox.showerror("Error", "Cannot Start Training While Preprocessing")
    else:
        global StopTrain
        global y
        global text
        global timer
        global sec
        global mint
        global hr
        StopTrain=False
        text.delete(1.0,tk.END)
        timer  = tk.Label(frame,text="",font=("Helvetica", 16))
        timer.place(x=0, y=70)
        sec=0
        mint=0
        hr=0
        update_clock()
        text.insert(tk.INSERT,"Started Training")
        y = threading.Thread(target=StartTraining,daemon=True)
        y.start()
    
global StopPre
StopPre=True
def preprocessdata():
    global z
    global text
    global timer
    global sec
    global mint
    global hr
    if(StopRec==False):
        messagebox.showerror("Error", "Cannot Start Preprocessing While Classroom Attendence is Running")
    elif(StopFram == False):
        messagebox.showerror("Error", "Cannot Start Preprocessing While Frames Are Extracting")
    elif(StopFace == False):
        messagebox.showerror("Error", "Cannot Start Preprocessing While Faces Are Extracting")
    elif(StopTrain == False):
        messagebox.showerror("Error", "Cannot Start Preprocessing While Training")
    else:
        global StopPre
        StopPre=False
        text.delete(1.0,tk.END)
        sec=0
        mint=0
        hr=0
        timer  = tk.Label(frame,text="",font=("Helvetica", 16))
        timer.place(x=0, y=70)
        update_clock()
        text.insert(tk.INSERT,"Started Pre-Processing Dataset")
        z = threading.Thread(target=StartPreprocess,daemon=True)
        z.start()

global sec
global mint
global hr

def update_clock():
    global sec
    global mint
    global hr

    sec+=1
    if(sec>59):
        sec=0
        mint+=1
    if(mint>59):
        mint=0
        hr+=1
    if(sec<10):
        S_sec = "0"+str(sec)
    else:
        S_sec = str(sec)
    if(mint<10):
        S_mint = "0"+str(mint)
    else:
        S_mint = str(mint)
    if(hr<10):
        S_hr = "0"+str(hr)
    else:
        S_hr = str(hr)
    now = "Timer: "+S_hr+" : "+S_mint+" : "+ S_sec
    #now = time.strftime("%H:%M:%S")
    timer.configure(text=now)
    root.after(1000, update_clock)

global timer
def startr():
    global x
    global text
    global StopRec
    global timer
    global sec
    global mint
    global hr
    if(StopFram == False):
        messagebox.showerror("Error", "Cannot Start Classroom Attendence While Frames are Extracting")
    elif(StopFace == False):
        messagebox.showerror("Error", "Cannot Start Classroom Attendence While Faces are Extracting")
    elif(StopTrain == False):
        messagebox.showerror("Error", "Cannot Start Classroom Attendence While Training")
    elif(StopPre == False):
        messagebox.showerror("Error", "Cannot Start Classroom Attendence While Preprocessing")
    else:
        StopRec = False
        start.lower()
        #stop.lift()
        text.delete(1.0,tk.END)
        timer  = tk.Label(frame,text="",font=("Helvetica", 16))
        timer.place(x=0, y=70)
        text.insert(tk.INSERT,"Starting Class Attendence")
        #text.pack()
        sec=0
        mint=0
        hr=0
        update_clock()
        x = threading.Thread(target=StartRecognition,daemon=True)
        x.start()





import glob

def FramesRead():
    print("Reading Images")
    images = [cv2.imread(file) for file in glob.glob("Frames/*.jpg")]
    print("Images Found: ",len(images))
    print("Started Face Detection")
    i=1
    for image in images:
        if(StopFace==True):
            text.insert(tk.INSERT,"\nCompleted...")
            print("Completed...")
            break
        faceDetect(image,i)
        i+=1











def streamRead():
    i=0
    video  = cv2.VideoCapture('Data_Set/00000.MTS')
    if (video.isOpened()== False): 
      print("Error opening video stream or file")
     
    else:
        count=0
        start1=time.time()
        while True:
            if(StopFram==True):
                print("Interrupted")
                text.insert(tk.INSERT,"Completed...")
                break;
            while(count<=150):
                ret,frame1 = video.read()
                count+=1
            count=0
            end1=time.time()
            text.delete(1.0,tk.END)
            if(ret):
                cv2.imwrite("Frames/Image-%04d.jpg"% i , frame1)
                i=i+1
                print(i)
                text.insert(tk.INSERT,"Frame Extracted: " +str(i))
                text.insert(tk.INSERT,"\n")
            print("time:",round(end1-start1,5))
        video.release()
    cv2.destroyAllWindows()



def frames():
    global w
    global text
    #global StopRec
    global timer
    global sec
    global mint
    global hr
    global StopFram
    if(StopRec == False):
        messagebox.showerror("Error", "Cannot Start Extracting While Classroom Attendence is Running")
    elif(StopFace == False):
        messagebox.showerror("Error", "Cannot Start Extracting While Faces Are Extracting")
    elif(StopTrain == False):
        messagebox.showerror("Error", "Cannot Start Extracting While Training")
    elif(StopPre == False):
        messagebox.showerror("Error", "Cannot Start Extracting While Preprocessing")
    else:
        StopFram = False
        frames.lower()
        #stop.lift()
        text.delete(1.0,tk.END)
        timer  = tk.Label(frame,text="",font=("Helvetica", 16))
        timer.place(x=0, y=70)
        text.insert(tk.INSERT,"Started Extracting Frames")
        #text.pack()
        sec=0
        mint=0
        hr=0
        update_clock()
        w = threading.Thread(target=streamRead,daemon=True)
        w.start()

global StopFram
StopRec = True
StopFram = True
def stopframes():
    global StopFram
    frames.lift()
    timer.destroy()
    #stop.lower()
    StopFram = True
    
def stopfaces():
    global StopFace
    faces.lift()
    timer.destroy()
    #stop.lower()
    StopFace = True

global StopFace
StopFace=True
def faces():
    global v
    global text
    #global StopRec
    global timer
    global sec
    global mint
    global hr
    global StopFace
    if(StopRec == False):
        messagebox.showerror("Error", "Cannot Extract Faces While Classroom Attendence is Running")
    elif(StopFram == False):
        messagebox.showerror("Error", "Cannot Extract Faces While Frames Are Extracting")
    else:
        StopFace = False
        faces.lower()
        #stop.lift()
        text.delete(1.0,tk.END)
        timer  = tk.Label(frame,text="",font=("Helvetica", 16))
        timer.place(x=0, y=70)
        text.insert(tk.INSERT,"Started Extracting Faces")
        #text.pack()
        sec=0
        mint=0
        hr=0
        update_clock()
        w = threading.Thread(target=FramesRead,daemon=True)
        w.start()
def stopr():
    global StopRec
    

    print(up_var.get())
    print(email_var.get())
    print(sms_var.get())
    start.lift()
    timer.destroy()
    #stop.lower()
    StopRec = True

def selectdir():
    filename = tk.filedialog.askdirectory()
    print(filename)


root = tk.Tk()
frame = tk.Frame(root,width=600,height=600)
Label = tk.Label(frame,text="FaceRacking Software",font=("Helvetica", 16,"bold"))
Label.place(x=200,y=20)

frames = tk.Button(frame,text="Frames From Stream",command= frames)
frames.place(x=420, y=70, height=30, width=150)
frame_stop = tk.Button(frame,text="Stop",command= stopframes)
frame_stop.place(x=420, y=70, height=30, width=150)
frames.lift()

faces = tk.Button(frame,text="Faces From Frames",command= faces)
faces.place(x=420, y=120, height=30, width=150)
faces_stop = tk.Button(frame,text="Stop",command= stopfaces)
faces_stop.place(x=420, y=120, height=30, width=150)
faces.lift()

preprocess = tk.Button(frame,text="Pre Process Data",command= preprocessdata)
preprocess.place(x=420, y=170, height=30, width=150)

train=tk.Button(frame, text ="Train Data",command = train)
train.place(x=420, y=220, height=30, width=150)

start=tk.Button(frame, text ="Start Class Attendence",command = startr)
start.place(x=420, y=270, height=30, width=150)
stop=tk.Button(frame, text ="Stop",command = stopr)
#stop.lower()

stop.place(x=420, y=270, height=30, width=150)
start.lift()
text=tk.Text(frame,font=("Helvetica", 16))
up_var = tk.IntVar()
email_var = tk.IntVar()
sms_var = tk.IntVar()
check = tk.Checkbutton(frame, text="Auto-Update", variable=up_var)
check.place(x=420, y=320)
email = tk.Checkbutton(frame, text="Auto-Email", variable=email_var)
email.place(x=420, y=370)
sms = tk.Checkbutton(frame, text="Auto-Sms", variable=sms_var)
sms.place(x=420, y=420)

text.place(x=0, y=120, height=450, width=400)
frame.pack()
#start.place(x=200,y=-100)
#select.pack()
#start.pack()
##train.pack()
#text.pack()
#stop.pack()
root.mainloop()



