from rnn import txtget as txt
import numpy as np

#튜닝
learning_rate = 0.001           #학습률
training_iters = 100000         #반복수
batch_size = 128                #미니배치 개수
display_step = 10               #로그추적 개수

#칫솔질 데이터
xfile=[]
yfile=[]

result = txt.get("02.g.txt", "02.a.txt", "02.t.txt")
for i in result:
    yfile.append(i[0])
    xfile.append(i[1:11])

dataset_x=np.array(xfile[0])
dataset_y=np.array(yfile)

print(dataset_x)