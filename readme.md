## Crawlers

### 네이트판 랭킹 웹 크롤러
세부 설명 : https://wdprogrammer.tistory.com/39?category=817328

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



## Preprocessor

정규표현식을 이용한 텍스트 노이즈 제거

**예시**

~~~python
from preprocessing import KoreanPreprocessor

processor = KoreanPreprocessor()
print(processor.remove_noises("[문자열]"))
~~~

입력

~~~
addLoadEvent(function(){j$('#contentVideo0').html('<iframe name=\"video\" class=\"cywriteVisualAid\" id=\"310100277\" src=\"//pann.nate.com/attach/videoPlay?attach_id=310100277&amp;attach_type=P&amp;pann_id=337265731\" frameborder=\"0\" scrolling=\"no\" style=\"width: 490px; height: 435px;\" swaf:cywrite:src=\"//pann.nate.com/attach/videoPlay?attach_id=310100277&amp;attach_type=P&amp;pann_id=337265731\" swaf:cywrite:attach_id=\"310100277\" swaf:cywrite:attach_type=\"P\" swaf:cywrite:video_id=\"20170528210501655197091005\" swaf:cywrite:video_nm=\"땡글이_헤테로폴리탄(heteropolitan).mp4\" swaf:cywrite:default_thumb_img=\"//mpmedia002.video.nate.com/img/005/DD/00/09/B_20170528210501655197091005.jpg\" swaf:cywrite:video_play_tm=\"247\" swaf:cywrite:thumb_url_1=\"//mpmedia002.video.nate.com/img/005/DD/00/09/B_20170528210501655197091005.jpg\" swaf:cywrite:thumb_tm_1=\"0\" swaf:cywrite:object_id=\"310100277\" allowfullscreen></iframe>');});\n \n 학교 폭력의 가해자들이 듣고 느끼는 바가 있었으면 하는 바람으로 만들었습니다.\n \n \n******************************************************************************\n톡커들의선택 1위된거 감사합니다. \n \n*******************************************************************************
~~~

출력

~~~
학교 폭력의 가해자들이 듣고 느끼는 바가 있었으면 하는 바람으로 만들었습니다.  톡커들의선택 1위된거 감사합니다.
~~~



## Utils

JSON 파일로 저장된 한글 문서 집합으로부터 TF-IDF 계산

~~~python
processor = KoreanPreprocessor()
dt = processor.get_segmented_docs(js_filename='nate_pann_ranking20210123_20210622_rm_noisy.json')
get_tfidf(dt['docs'], dt['titles'], n_limit=300).to_csv('tf_idf.csv', index=False, encoding="utf-8-sig")
~~~

결과 예시

| word     | freq | 유명 커플 유튜버, 극단적 선택 시도..성관계 영상 … | 고백썰 풀어주라! | 딩크 부부의 고민. 제 잘못인가요? | 에이프릴 또 거짓말함 | 빅히트 사내연애 아님 하이브임 방셉 | 근데 트와이스는 재계약 절대안할거같음... |
| -------- | ---- | ------------------------------------------------- | ---------------- | -------------------------------- | -------------------- | ---------------------------------- | ---------------------------------------- |
| 있       | 6    | 4.925883                                          | 3.283922         | 7.388825                         | 0                    | 1.641961                           | 0.820981                                 |
| 영상     | 1    | 39.86775                                          | 0                | 0                                | 0                    | 0                                  | 0                                        |
| 시어머니 | 11   | 0                                                 | 0                | 3.506558                         | 0                    | 0                                  | 0                                        |
| 기       | 2    | 2.746098                                          | 2.746098         | 6.865246                         | 0                    | 1.373049                           | 1.373049                                 |