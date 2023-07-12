import cv2
import glob
import lbp
import findface
import matplotlib.pyplot as plt

path = "att/*"
data = []
files = glob.glob(path)
for file in files:
    image = cv2.imread(file)
    data.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
query = cv2.cvtColor(cv2.imread('19.png'),cv2.COLOR_BGR2RGB)
# check if there is any face in picture???????????????
flag , _ = findface.face(query)
if flag==False:
    raise Exception("SORRY there is no face in your image")

C = lbp.Comparator(data,query)
output1 = C.lbp()

f = plt.figure(figsize=(15,15))
f.add_subplot(1,2,1)
plt.imshow(query)
plt.title('Input image')
f.add_subplot(1,2,2)
plt.imshow(output1)
plt.title('Closest Image By LBP')
plt.show()