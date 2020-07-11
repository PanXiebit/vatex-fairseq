import nltk
from tqdm import tqdm

#
# for split in ["train", "valid"]:
#     with open("Data/separate_data/%s.vid-en.en"%split, "r") as f, \
#             open("Data/vid-nnen/%s.vid-nnen.nnen"%split, "w") as fw:
#         for i, line in tqdm(enumerate(f)):
#             text = line.strip().split()
#             pos_tagged = nltk.pos_tag(text)
#             cur_nn = []
#             for word, pos in pos_tagged:
#                 if pos in ["NN", "NNS", "NNP", "NNPS"]:
#                     # if not en_d.check(word):
#                     #     correct_word = en_d.suggest(word)
#                     #     line.replace(word, correct_word[0])
#                     #     print("The word {}/{} in '{}'  is wrong".format(word, correct_word[0], line))
#                     #     cur_nn.append(correct_word[0])
#                     # else:
#                     cur_nn.append(word)
#             if len(cur_nn) == 0:
#                 print(line)
#             fw.write(" ".join(cur_nn) + " ||| " + line)
#
#
from collections import defaultdict
nn_dict = defaultdict(int)
tgt_dict = defaultdict(int)
#
with open("Data/vid-nnen/train.vid-nnen.nnen", "r") as f:
    for i, line in enumerate(f):
        content = line.strip().split(" ||| ")
        assert len(content) == 2
        nn_text = content[0].strip().split()
        tgt_text = content[1].strip().split()
        for word in nn_text:
            nn_dict[word] += 1
        for word in tgt_text:
            tgt_dict[word] += 1

sorted_nn_dict = sorted(nn_dict.items(), key=lambda item: item[1], reverse=True)
sorted_tgt_dict = sorted(tgt_dict.items(), key=lambda item: item[1], reverse=True)
print("nn vocabulary size: ", len(nn_dict))  # 18170  好多错别字
print("tgt vocabulry size: ", len(tgt_dict)) # 27147

with open("Data/vid-nnen/dict.nn.txt", "w") as f:
    for word, cnt in sorted_nn_dict:
        f.write(word + " " + str(cnt) + "\n")

with open("Data/vid-nnen/dict.nnen.txt", "w") as f:
    for word, cnt in sorted_tgt_dict:
        f.write(word + " " + str(cnt) + "\n")

