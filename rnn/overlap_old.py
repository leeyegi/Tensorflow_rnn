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

        #aglist.append([i, aat, gat])  # 순서번호 포함
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
N = 50  # 하나의 데이터 셋에 요소 개수
ol = 50  # overlap 배율

def get(afile, gfile, tfile,ti):
    aglist = mkaglist(afile, gfile)
    tlist = mktlist(tfile)
    result = []
    testresult = []


    for i in tlist:  # 클래스 만큼 반복
        cntft = 0;
        tmplist = []
        s = int(i[1]) #클래스 시작점
        e = int(i[2]) #클래스 끝점
        rs = int((e-s)/5)
        testrange = (s+rs*(ti-i)+1,s+rs*ti)
        j = s  # 클래스의 시작점부터
        while (j <= e):  #클래스의 끝점까지
            if cntft == N:  # N개 만큼 tmp리스트에 채웠다면
                if testrange[0] < j and testrange[1] < j - N - ol:  # j가 현재 반복인덱스 줄이기전 인덱스, overlap제외범위 시작보다 크면 인덱스 안줄임, overlap제외범위 끝보다 줄인 값이 작으면 안줄임
                    testresult.append(tmplist)
                    j -= int(N - ol)
                else:
                    result.append(tmplist)  # 결과리스트에 추가
                    j -= int(N - ol)  # overlap (반복인덱스를 줄임)
                cntft = 0  # 한 데이터셋 갯수카운트 초기:
                tmplist = []  # 결과리스트 초기화
            if cntft == 0:  # 첫번째 데이터앞은 클래스 정보추가
                tmplist.append(i[0])
            tmplist.append(aglist[j])
            cntft += 1
            j += 1

    return result,testresult

# getmany([[1,2,3],[1,2,3],[1,2,3]])
def getmany(nlist,ti):
    result = []
    testresult = []
    for i in nlist:
        a,b = get(i[0],i[1],i[2],ti)
        result+= a
        testresult += b
    random.shuffle(result)
    return result,testresult



if __name__ == "__main__":
    # l = mkaglist("03.a.txt","03.g.txt")
    l = get('dataset/03.a.txt', 'dataset/03.g.txt', "dataset/03.t.txt")
    for i in l:
        print(i[0])