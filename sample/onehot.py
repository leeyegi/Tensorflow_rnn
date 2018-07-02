import pandas as pd
from pandas import DataFrame, Series
from itertools import cycle


pd.set_option('display.unicode.east_asian_width', True)

area = '서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주'.split()
college = '인문대학 사회과학대학 자연과학대학 간호대학 경영대학 공과대학 미술대학'.split()
gender = '남성 여성'.split()

# 100개의 가짜 데이터 생성, itertools.cycle 함수로 각 요소를 순환시킵니다.
fake_data = zip(range(100), cycle(area), cycle(college), cycle(gender))
hundred_students = DataFrame([data for num, *data in fake_data],
                              columns='지역 단과대 성별'.split())
hundred_students.head(10)

# pd.get_dummies로 One-hot 인코딩
college_one_hot_encoded = pd.get_dummies(hundred_students.단과대)

# 원래 데이터와 비교식으로 보여주기용 데이터프레임
college_with_onehot = pd.concat(
	[DataFrame(hundred_students.단과대), college_one_hot_encoded],
	axis=1)
college_with_onehot.head(10)

pd.get_dummies(hundred_students, prefix=['지역', '단과대', '성별']).head(10)


