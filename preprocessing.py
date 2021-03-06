from konlpy.tag import Mecab
# from soynlp.word import WordExtractor
import json, re

# Segment
class KoreanPreprocessor(object):
    def __init__(self):
        self.mecab = Mecab()
        
    def remove_noises(self, doc, strict=True):
        # print(doc)
        # Remove video tag
        doc = re.sub(r'(addLoadEvent)[^\n]+;', "", doc)

        # Remove line alignments
        doc = re.sub(r'[\n\s]+', " ", doc)

        # Remove links
        doc = re.sub(r'(http)([s:/]{3,4})?[a-z0-9]+([\.][a-z0-9]+)+[\S]+', "", doc)

        # Remove repetition
        if strict:
            doc = re.sub(r'[ㄱ-ㅎㅏ-ㅣ\s]+', ' ', doc)
        else:
            doc = re.sub(r'([ㄱ-ㅎㅏ-ㅣ\.\s])\1{2,}', r'\1\1', doc)
        doc = re.sub(r'([^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s\.]){2,}', r'\1', doc)
        
        # Remove special char
        doc = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s,/\?\!\.]', '', doc)

        return doc

    def remove_empty_docs_and_noises(self, docs=None, js_filename=None, strict=True, to_json=True):
        if js_filename is not None:
            with open(js_filename, 'r') as js:
                docs = json.load(js)
        
        renew_docs = []
        for idx in range(len(docs)):
            doc = self.remove_noises(docs[idx]['content'], strict=strict)
            if len(doc) != 0:
                docs[idx]['title'] = self.remove_noises(docs[idx]['title'])
                docs[idx]['content'] = doc
                renew_docs.append(docs[idx])

        print('The number of removed documents', len(docs) - len(renew_docs))
        if to_json:
            with open(js_filename[:-5]+'_rm_noisy.json', 'w') as js:
                json.dump(renew_docs, js, ensure_ascii=False)

        return renew_docs

    def segment(self, doc):
        doc = self.mecab.morphs(doc)
        return doc

    def get_segmented_docs(self, docs=None, js_filename=None):
        if js_filename is not None:
            with open(js_filename, 'r') as js:
                docs = json.load(js)
        
        titles = []
        seg_docs = []
        for doc in docs:
            titles.append(doc['title'])
            seg_docs.append(self.segment(doc['content']))

        return {'titles':titles, 'docs': seg_docs}
    

if __name__=='__main__':
    processor = KoreanPreprocessor()
    processor.remove_empty_docs_and_noises(js_filename='nate_pann_ranking20210123_20210622.json', to_json=True)
    
    # print(processor.remove_noises("addLoadEvent(function(){j$('#contentVideo0').html('<iframe name=\"video\" class=\"cywriteVisualAid\" id=\"310100277\" src=\"//pann.nate.com/attach/videoPlay?attach_id=310100277&amp;attach_type=P&amp;pann_id=337265731\" frameborder=\"0\" scrolling=\"no\" style=\"width: 490px; height: 435px;\" swaf:cywrite:src=\"//pann.nate.com/attach/videoPlay?attach_id=310100277&amp;attach_type=P&amp;pann_id=337265731\" swaf:cywrite:attach_id=\"310100277\" swaf:cywrite:attach_type=\"P\" swaf:cywrite:video_id=\"20170528210501655197091005\" swaf:cywrite:video_nm=\"땡글이_헤테로폴리탄(heteropolitan).mp4\" swaf:cywrite:default_thumb_img=\"//mpmedia002.video.nate.com/img/005/DD/00/09/B_20170528210501655197091005.jpg\" swaf:cywrite:video_play_tm=\"247\" swaf:cywrite:thumb_url_1=\"//mpmedia002.video.nate.com/img/005/DD/00/09/B_20170528210501655197091005.jpg\" swaf:cywrite:thumb_tm_1=\"0\" swaf:cywrite:object_id=\"310100277\" allowfullscreen></iframe>');});\n \n 학교 폭력의 가해자들이 듣고 느끼는 바가 있었으면 하는 바람으로 만들었습니다.\n \n \n******************************************************************************\n톡커들의선택 1위된거 감사합니다. \n \n*******************************************************************************"))
        
