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


