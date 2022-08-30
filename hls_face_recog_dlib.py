import face_recognition as face
import numpy as np
import cv2
import psutil
import time
from datetime import timedelta
import sys

start_time = time.time()


#ORIGINAL_CODE_CREDIT:  https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

#******#กด q เพื่อออกจากหน้าต่าง**********
#รับ video --> ใส่ path, webcam --> ใส่ 0
# .m3u8          = cant pass
# HTTP URL m3u8 = pass 
# .ts           = pass
VIDEO_URL = "http://cdnapi.kaltura.com/p/1878761/sp/187876100/playManifest/entryId/1_usagz19w/flavorIds/1_5spqkazq,1_nslowvhp,1_boih5aji,1_qahc37ag/format/applehttp/protocol/http/a.m3u8"
video_capture = cv2.VideoCapture(VIDEO_URL)
cap = cv2.VideoCapture(VIDEO_URL)
if (cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)
# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)
# test HLS
# 


# รูปภาพที่จะใช้เทียบ------------------------------------------------------
def LoadFaces():
# Jennie
    jennie_image = face.load_image_file("photo/Jennie.jpg")
    jennie_face_encoding = face.face_encodings(jennie_image)[0]
# Lisa
    lisa_image = face.load_image_file("photo/Lisa.jpg")
    lisa_face_encoding = face.face_encodings(lisa_image)[0]
# Rose
    rose_image = face.load_image_file("photo/rose.jpg")
    rose_face_encoding = face.face_encodings(rose_image)[0]
# Jisoo
    jisoo_image = face.load_image_file("photo/jisoo.jpg")
    jisoo_face_encoding = face.face_encodings(jisoo_image)[0]
    
    known_face_encodings = [jennie_face_encoding,lisa_face_encoding, rose_face_encoding, jisoo_face_encoding]
    known_face_names = ["JENNIE", "LISA", "ROSE", "JISOO"]

    return known_face_encodings, known_face_names;

#ตัวแปรข้อมูล--------------------------------------------------------------
face_locations = [] 
face_encodings = []
face_names = []
face_percent = []

# process frame by frame
process_this_frame = True

known_face_encodings, known_face_names = LoadFaces()

#loopคำนวณแต่ละเฟรมของวิดีโอ
while True:
    #อ่านค่าแต่ละเฟรมจากวิดีโอ
    ret, frame = video_capture.read()
    if ret:
        #ลดขนาดสองเท่าเพื่อเพิ่มfps 
        small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
        #เปลี่ยน bgr เป็น rgb 
        rgb_small_frame = small_frame[:,:,::-1]

        face_names = []
        face_percent = []

        if process_this_frame:
            #ค้นหาตำแหน่งใบหน้าในเฟรม โดยมี model 2อย่างคือ cnn, hog
            #cnn จะเป็น modelรุ่นใหม่แม่นยำกว่าแต่กินทรัพยากรมาก
            face_locations = face.face_locations(rgb_small_frame, model="hog")
            #นำใบหน้ามาหาfeaturesต่างๆที่เป็นเอกลักษณ์ (ตา, จมูก, ปาก,...)
            face_encodings = face.face_encodings(rgb_small_frame, face_locations)
            
            #เทียบแต่ละใบหน้า
            for face_encoding in face_encodings:
                face_distances = face.face_distance(known_face_encodings, face_encoding)
                #หาตำแหน่งที่ดีที่สุดบนระยะใบหน้า
                best = np.argmin(face_distances)
                face_percent_value = 1-face_distances[best]

                #กรองใบหน้าที่ความมั่นใจ50% ปล.สามารถลองเปลี่ยนได้
                if face_percent_value >= 0.5:
                    name = known_face_names[best]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                else:
                    name = "UNKNOWN"
                    face_percent.append(0)
                face_names.append(name)

        #วาดกล่องและtextเมื่อแสดงผลออกมาออกมา
        for (top,right,bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top*= 2
            right*= 2
            bottom*= 2
            left*= 2

            if name == "UNKNOWN":
                color = [46,2,209]
            else:
                color = [255,102,51]
            #กรอบdetectใบหน้า
            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left-1, top -30), (right+1,top), color, cv2.FILLED)
            cv2.rectangle(frame, (left-1, bottom), (right+1,bottom+30), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)

        #สลับค่าเป็นค่าตรงข้ามเพื่อให้คิดเฟรมเว้นเฟรม
        process_this_frame = not process_this_frame

        #แสดงผลลัพท์
        cv2.imshow("Video", frame)
        #กด q เพื่อออกจากหน้าต่าง
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break

    else:
        break

#Execution time
elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)
print('The CPU usage is: ', psutil.cpu_percent(4))
print('RAM memory % used:', psutil.virtual_memory()[2])
    
#ล้างค่าต่างๆเมื่อปิดโปรแกรม
video_capture.release()
cv2.destroyAllWindows()


