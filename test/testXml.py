from xml.etree.ElementTree import Element, SubElement, dump, ElementTree
import glob
from PIL import Image
#파일안에 이미지 모두 불러와 xml file generate

#이미지 경로
images = glob.glob('C:\\Users\yegilee\Documents\dataxml\\tot/*.jpg')
print(images)

for fname in images:
    #이미지 불러오고 사이즈 저장
    im = Image.open(fname)
    (w, h) = im.size

    #구분자 찾기
    #이미지 파일은 순서번호_ROIWidth+ROIHeight.jpg로 구성됨
    searchNo=fname.find('tot\\')     #이름 앞
    searchNo2=fname.find('_')       #이름 뒤 + width 앞
    searchNo3=fname.find('+')       #width 뒤 + height 앞
    searchNo4=fname.find('.jpg')    # height 뒤 구분자들 위치 찾기
    if searchNo2==-1 or searchNo3==-1:
        continue
    '''
    print(searchNo)
    print(searchNo2)
    print(searchNo3)
    print(searchNo4)

    print(fname[searchNo+3:searchNo2])
    print(fname[searchNo2+1:searchNo3])
    print(fname[searchNo3+1:searchNo4])
    '''
    #xml 파일 이름
    xml_title =fname[searchNo+4:-4]            #파일타이틀이름
    annotation = Element("annotation")

    folder = SubElement(annotation, "folder").text = 'JPEGImages'
    filename = SubElement(annotation, "filename").text = fname[searchNo+4:]     #파일이름

    path = SubElement(annotation, "path").text = 'C:\\Users\yegilee\Documents\dataxml\\tot'

    source = SubElement(annotation, "source")
    database = SubElement(source, "database").text = 'Unknown'

    size = SubElement(annotation, "size")
    #image size
    width = SubElement(size, "width").text = str(w)
    height = SubElement(size, "height").text = str(h)
    depth = SubElement(size, "depth").text = '3'

    segmented = SubElement(annotation, "segmented").text = '0'
    object = SubElement(annotation, "object")

    #check
    name = SubElement(object, "name").text = 'tot'
    pose = SubElement(object, "pose").text = 'Unspecified'
    truncated = SubElement(object, "truncated").text = '0'
    difficult = SubElement(object, "difficult").text = '0'

    #roi
    bndbox = SubElement(object, "bndbox")
    xmin = SubElement(bndbox, "xmin").text = fname[searchNo2+1:searchNo3]
    ymin = SubElement(bndbox, "ymin").text = fname[searchNo3+1:searchNo4]
    xmax = SubElement(bndbox, "xmax").text = str(int(fname[searchNo2+1:searchNo3]) + 10)
    ymax = SubElement(bndbox, "ymax").text = str(int(fname[searchNo3+1:searchNo4]) + 10)



    #print("list : {0}".format(i))

    dump(annotation)
    ElementTree(annotation).write("C:\\Users\yegilee\Documents\dataxml\\tot\\xml\{0}.xml".format(xml_title))
