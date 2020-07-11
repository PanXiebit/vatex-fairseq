import os
import json
from tqdm import tqdm
import re
from data_preprocess import bert_tokenization

# def ch_split(s1):
#     # 把句子按字分开，中文按字分，英文按单词，数字按空格
#     regEx = re.compile('[\\W]*')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
#     res = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5]中文范围
#
#     p1 = regEx.split(s1.lower())
#     str1_list = []
#     for str in p1:
#         if res.split(str) == None:
#             str1_list.append(str)
#         else:
#             ret = res.split(str)
#             for ch in ret:
#                 str1_list.append(ch)
#
#     list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
#
#     return " ".join(list_word1)
#
# def en_split(en_sents):
#     return " ".join(word_tokenize(en_sents))


basic_tokenize = bert_tokenization.BasicTokenizer()
def tokenize(sent):
    return " ".join(basic_tokenize.tokenize(sent))

if __name__ == "__main__":
    for split in ["train", "valid"]:
        if split == "train":
            corpus = "/root/VatexData/vatex_training_v1.0.json"
        elif split == "valid":
            corpus = "/root/VatexData/vatex_validation_v1.0.json"
        else:
            corpus = None
        with open(corpus, "r") as f:
            content = json.load(f)
            print(len(content))
        vid_paths = "Data/separate_data/{}.vid-en.vid".format(split)
        en_paths = "Data/separate_data/{}.vid-en.en".format(split)
        ch_paths = "Data/separate_data/{}.vid-ch.ch".format(split)
        with open(vid_paths, "w") as fw1, open(en_paths, "w") as fw2, open(ch_paths, "w") as fw3:
            for i, text in tqdm(enumerate(content)):
                assert "enCap" in text and "chCap" in text
                en_sents = text["enCap"]
                ch_sents = text["chCap"]
                assert len(en_sents) == len(ch_sents) == 10
                vid_path = os.path.join("/root/VatexData/trainval", text["videoID"])
                for j in range(10):
                    fw1.write(vid_path + "\n")
                    fw2.write(tokenize(en_sents[j].lower()) + "\n")
                    fw3.write(tokenize(ch_sents[j]) + "\n")
