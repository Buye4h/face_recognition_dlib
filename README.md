# face_recognition_dlib

ติดตั้ง pip ใน terminal
'"
pip install cmake
pip3 install face-recognition
pip3 install numpy
pip3 install opencv-python
'"

ปัญหา pylint ในกรณี VS Code ไม่พบ cv2 หลังติดตั้ง opencv-python
[https://webnautes.tistory.com/1674]

On Windows/Linux - File > Preferences > Settings
On macOS - Code > Preferences > Settings
search --> pylint
ใส่ข้อความดังนี้ใน pylint: Args

--extension-pkg-whitelist=cv2
--generate-members
--disable=C0111
