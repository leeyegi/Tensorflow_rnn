from xml.etree.ElementTree import Element, SubElement, dump, ElementTree
import glob
from PIL import Image
import os
#해당 파일의 모든 이미지 불러와서 순서번호로 바꾸는 파일

#이미지 경로
images = glob.glob('C:\\Users\yegilee\Documents\dataxml\\tot/*.jpg')
print(images)
i=0
for fname in images:
    #+n은 마지막 폴더명 길이 +1
    searchNo=fname.find('tot\\')

    print(fname[:searchNo+4])

    #파일 읽어오는 순서대로 파일명을 i(순서번호)로 바꿈
    os.rename(fname,fname[:searchNo+4]+"/"+str(i)+".jpg")
    print("ok")
    i+=1