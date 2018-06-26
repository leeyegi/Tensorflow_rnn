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
ol = 10  # overlap 배율
'''
N = 50  # 하나의 데이터 셋에 요소 개수
ol = 10  # overlap 배율

def get(afile, gfile, tfile):
    aglist = mkaglist(afile, gfile)
    tlist = mktlist(tfile)
    result = []


    for i in tlist:  # 클래스 만큼 반복
        cntft = 0;
        tmplist = []
        j = int(i[1])  # 한 클래스의 시작점부터
        while (j <= int(i[2])):  # 한 클래스의 끝점까지
            if cntft == N:  # N개 만큼 tmp리스트에 채웠다면
                result.append(tmplist)  # 결과리스트에 추가
                tmplist = []  # 결과리스트 초기화
                cntft = 0  # 한 데이터셋 갯수카운트 초기화
                j -= int(N - ol)  # overlap (반복인덱스를 줄임)
            if cntft == 0:  # 첫번째 데이터앞은 클래스 정보추가
                tmplist.append(i[0])
            tmplist.append(aglist[j])
            cntft += 1
            j += 1
    return result

# getmany([[1,2,3],[1,2,3],[1,2,3]])
def getmany(nlist):
    result = []
    for i in nlist:
        result += get(i[0],i[1],i[2])
    random.shuffle(result)
    return result



if __name__ == "__main__":
    # l = mkaglist("03.a.txt","03.g.txt")
    l = get('dataset/03.a.txt', 'dataset/03.g.txt', "dataset/03.t.txt")
    for i in l:
        print(i[0])