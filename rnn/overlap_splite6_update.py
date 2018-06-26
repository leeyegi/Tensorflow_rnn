import random

def mkaglist(afile, gfile):
    '''[[자이로],[가속도]] 가 원소인 list 반환'''
    glines = open(gfile, "r").readlines()  # 자이로 파일
    alines = open(afile, "r").readlines()  # 가속도 파일
    aglist = []  # 자이로데이터,가속도 데이터담을 list
    for i in range(len(alines)):
        aat = alines[i].split(" ")[1:4]  # 가속도 xyz 1 set
        gat = glines[i].split(" ")[1:4]  # 자이로 xyz 1 set
        for j in aat:
            j = float(j)
        for j in gat:
            j = float(j)
        #aglist.append([[]])  # 순서번호,data 미포함
        #aglist.append([i, []])  # 순서번호 포함
        aglist.append(aat+gat) #순서번호 미포함
    return aglist


def mktlist(tfile):
    '''[class, 시작점, 끝점] 가 원소인 list 반환'''
    tlines = open(tfile, "r").readlines()  # 태그 파일
    # tagfile to list
    tlist = []
    for i in tlines:
        tlist.append(i.split(" ")[0:3])
    return tlist

'''
N = 50  # 하나의 데이터 셋에 요소 개수
ol = 5  # overlap 배율
'''
N = 50  # 데이터 셋 하나의 요소 개수
ol = 10  # overlap 배율

def get(afile, gfile, tfile):
    aglist = mkaglist(afile, gfile)
    tlist = mktlist(tfile)
    rangelist=[] # subset range list of all class
    seglist=[[] for i in range(6)]
    
    for i in tlist:  # 클래스 만큼 반복
        i[0] = int(i[0]) # 한 클래스 start 
        i[1] = int(i[1]) # 한 클래스 end
        i[2] = int(i[2]) # 클래스 name
        size = int((i[2] - i[1]) / 6) #한클래스범위 1/6 size (subset size)
       # print("size" ,size)
        tmprangelist=[] #한 클래스의 서브셋 범위 리스트
        for k in range(6):
            tmprangelist.append([i[1],i[1]+size-1,i[0]])
            i[1]=i[1]+size
        rangelist.append(tmprangelist)

###################
    '''list23 = [ 0 for i in range(6)]
    for i in rangelist:
        #print(i)
        nsum = 0
        for j in range(len(i)):
            list23[j] += i[j][1]- i[j][0]

    print(list23)'''
################

        
    for i in rangelist: # class갯수번
        for j in range(6): # 6번 (클래스당 6번)
            result=[]
            k = i[j][0]  # 한 클래스 subset의 시작점부터
            cntft= 0 #데이터셋 인덱스 카운트
            tmplist = [] #데이터 한개 리스트
            while (k <= i[j][1]):  # 한클래스 subset의 끝점까지
                if cntft == N:  # N개 만큼 tmplist에 채우면
                    result.append(tmplist)  # 서브셋 한개 완성 / 추가
                    tmplist = []  # 결과리스트 초기화
                    cntft = 0  # 한 데이터셋 갯수카운트 초기화
                    k -= int(N - ol)  # overlap (반복인덱스를 줄임)
                if cntft == 0:  # 첫번째 데이터앞은 클래스 정보추가
                    tmplist.append(i[j][2]) 
                    #print(j,j.__class__)
                tmplist.append(aglist[k])#데이터 1개
                cntft += 1
                k += 1
            #print(j,len(result))
            seglist[j] = seglist[j]+result #j번째 subset에 데이터 추가
    return seglist

# getmany([[1,2,3],[1,2,3],[1,2,3]])
def getmany(nlist):
    result = [[] for i in range(6)]

    for i in nlist:
        getd = get(i[0],i[1],i[2])
        #for j in range(5):
        for j in range(6):
            result[j] += getd[j]
    for i in result:
        random.shuffle(i)
    random.shuffle(result)
    return result



if __name__ == "__main__":
    # l = mkaglist("03.a.txt","03.g.txt")
    l = get('a.txt', 'g.txt', "t.txt")
    outfile = open("outfile.txt", "w")
    p=0
    for i in l:
        print(len(i))
        for j in i:
            #print(len(j))
            outfile.write(str(j)+"\n")

    outfile.close()

        
