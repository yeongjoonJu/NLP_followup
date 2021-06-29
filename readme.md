## Word2Vec

Skip Gram의 성능을 향상시키기 위한 negative sampling, subsampling 등을 제안한 [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)를 구현

**Training skip gram model**

word2vec.py

~~~python
train_skip_gram()
~~~

*Subsampling of Frequent words 구현에 대한 언급*

1. subsampling의 확률 값을 구한 뒤, 100 epochs을 한다고 산정하였을 때 $확률 \times 100$ 만큼 각 sample을 증대시킴

학습속도 향상을 위해 위 방법을 시도했으나 메모리를 너무 많이 잡아먹어 좋지 않은 방법이었음.

2. subsampling의 확률 값을 구한 뒤, 각 sample마다 해당 확률을 부여하고 WeightedSampler를 사용

메모리 문제는 해결되나 학습 속도가 많이 느려짐

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

Nate pann 랭킹 문서들로부터 단어 사전 만들기

~~~python
# Make vocabulary based Nate pann ranking documents
processor = KoreanPreprocessor()
data = processor.read_docs_from_js('data/nate_pann_ranking20210123_20210622.json')
docs = []
for sample in data:
    docs.append(sample['content'])

    vocab = get_vocab(docs, vocab_size=5000, rm_stopwords=True)
    with open('vocab.js', 'w') as js:
        json.dump(vocab, js, ensure_ascii=False)
~~~



JSON 파일로 저장된 한글 문서 집합으로부터 TF-IDF 계산

~~~python
processor = KoreanPreprocessor()
dt = processor.get_segmented_docs(js_filename='nate_pann_ranking20210123_20210622_rm_noisy.json')
get_tfidf(dt['docs'], dt['titles'], n_limit=50).to_csv('tf_idf.csv', index=False, encoding="utf-8-sig")
~~~

결과 예시

| word | freq | 유명 커플 유튜버, 극단적 선택 시도..성관계 영상 ... | 고백썰 풀어주라! | 딩크 부부의 고민. 제 잘못인가요? | 에이프릴 또 거짓말함 | 빅히트 사내연애 아님 하이브임 방셉 | 근데 트와이스는 재계약 절대안할거같음... | 에이프릴 멤버들 인터뷰 뜸 | 새우튀김 1개 환불건 때문에 점주 뇌출혈 사망 | 방탄 진 콧대 장난아니다 | 에스파 축하해 레전드다 | 에스파 스엠 최초 1위래 | 이번에 교사 뽑는 체제가 바뀌어.. 꼭 봐줘 |
| ---- | ---- | --------------------------------------------------- | ---------------- | -------------------------------- | -------------------- | ---------------------------------- | ---------------------------------------- | ------------------------- | ------------------------------------------- | ----------------------- | ---------------------- | ---------------------- | ---------------------------------------- |
| 생각 | 32   | 1.139434                                            | 0                | 9.115474                         | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 0                                        |
| 다고 | 31   | 13.62715                                            | 0                | 9.084766                         | 0                    | 0                                  | 0                                        | 1.514128                  | 1.514128                                    | 0                       | 0                      | 0                      | 0                                        |
| 같   | 31   | 0.867501                                            | 1.735001         | 3.470002                         | 0                    | 0.867501                           | 0.867501                                 | 0                         | 0                                           | 0                       | 0                      | 0                      | 0.867501                                 |
| 친구 | 30   | 7.330326                                            | 0                | 1.832581                         | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 0                                        |
| 겠   | 30   | 0                                                   | 0                | 8.91076                          | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 1.272966                                 |
| 으로 | 30   | 2.545931                                            | 0                | 5.091863                         | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 0                                        |
| 아니 | 29   | 3.064954                                            | 0                | 2.043302                         | 0                    | 0                                  | 1.021651                                 | 0                         | 0                                           | 0                       | 0                      | 0                      | 1.021651                                 |
| 아   | 29   | 0                                                   | 2.902752         | 0.967584                         | 0                    | 0                                  | 0.967584                                 | 0                         | 0.967584                                    | 0                       | 0                      | 0                      | 0.967584                                 |
| 살   | 28   | 1.514128                                            | 0                | 9.084766                         | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 0                                        |
| 씨   | 27   | 70.33527                                            | 0                | 0                                | 0                    | 0                                  | 0                                        | 0                         | 0                                           | 0                       | 0                      | 0                      | 0                                        |
| 해서 | 27   | 0                                                   | 0                | 3.236429                         | 0                    | 0                                  | 0                                        | 1.07881                   | 0                                           | 0                       | 0                      | 0                      | 4.315239                                 |