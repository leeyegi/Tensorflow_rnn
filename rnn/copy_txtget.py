import random

def get(gfile,afile,tfile):
    glines = open(gfile,"r").readlines()
    alines = open(afile,"r").readlines()
    tlines = open(tfile,"r").readlines()

    s = []
    e = []

    for i in tlines:
        t = i.split(";")
        s.append(int(t[1]))
        if t[2][-1]=="\n":
            e.append(int(t[2][0:-1]))
        else:
            e.append(int(t[2]))
        
    n = 0
    glist = []
    alist = []

    for i in glines:   
        tlist = []
        tstr=""
        chk = 0
        for j in i:
            if j == "\t":
                chk = 1
            if chk:
                if j!="\t" and j!="\n":
                    tstr += j
                elif tstr != "":
                    tlist.append(float(tstr))
                    tstr=""
        glist.append(tlist)
        tlist=[]

    for i in alines:
        tlist = []
        tstr=""
        chk = 0
        for j in i:
            if j == "\t":
                chk = 1
            if chk:
                if j!="\t" and j!="\n":
                    tstr += j
                elif tstr != "":
                    tlist.append(float(tstr))
                    tstr=""
        alist.append(tlist)
        tlist=[]


    result = []
    tre = []
    tre2 = []
    n=0

    for i in range(0,len(alist)):
        for j in range(16):
            if s[j]-1 <= i and i <=e[j]-1:
                if i == s[j]-1:
                    tre=[]
                    n=0  
                if n==0: 
                    tre.append(j+1)  
                tre2.append(alist[i])
                tre2.append(glist[i])
                tre.append(tre2)
                tre2=[]
                n+=1
                if n==10:
                    result.append(tre)
                    tre = []
                    n=0


    #random.shuffle(result)

    for i in result:
        for j in range(11):
            if j !=0:
                i[j] =  i[j][0]+i[j][1]
                


    return result


# getmany([[1,2,3],[1,2,3],[1,2,3]])
def getmany(nlist):
    result = []
    for i in nlist:
        result += get(i[0],i[1],i[2])

    random.shuffle(result)
    return result

        

if __name__ == "__main__":
    #l = getmany([["02.g.txt","02.a.txt","02.t.txt"]])
    l = getmany([["1g2018041952.txt", "1a2018041952.txt", "1t2018041952.txt"],
                          ["2g201804091947.txt", "2a201804091947.txt", "2t201804091947.txt"],
                          ["3g20180409131241.txt", "3a20180409131241.txt", "3t20180409131241.txt"]])

    cn=0
    for i in l:
        cn+=1
        print(i)

    print(cn)

