import urllib.request
from bs4 import BeautifulSoup

#쿼리받아서 크롤링 수행, URL 생성하는 메소드
def returnURL(query):
    #검색해야할 query를 퍼센트 인코딩으로 이스케이프 해줌
    #한글 또는 영어로 바로 검색하면 오류난대여
    query = urllib.parse.quote(query)

    #youtube에 해당 query값 추가 해 저장
    url = "https://www.youtube.com/results?search_query=" + query

    '''
    #요청 할 url보내고 받음
    response = urllib.request.urlopen(url)

    #response 읽어옴
    html = response.read()

    #크롤링라이브러리를 통해 유의미한 데이터 뽑음
    soup = BeautifulSoup(html, 'html.parser')

    #모든 값 뽑아옴
    #URLList=[]
    for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
        URL='https://www.youtube.com/results?search_query=' + vid['href']
    '''
    return url

if __name__ == "__main__":
    URL = returnURL('추신수 경기')
    print(URL)