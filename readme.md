## Crawlers

### 네이트판 랭킹 웹 크롤러

crawlers.py

~~~python
crawler = NatePannRankingCrawler()
"""
days : 며칠동안의 랭킹 글을 크롤링할 것인지 설정
current_time : 실시간 랭킹 글을 포함 (전체, 일상톡, 10대톡, 연예톡)
to_json : json 파일로 저장
"""
crawler.get_contents(days=150, current_time=True, to_json=True)
~~~