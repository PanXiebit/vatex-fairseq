
from collections import defaultdict

train_corpus = "Data/bpe_data/train.vid-ench.ench"
ench_dict_file = "Data/bpe_data/dict.ench.txt"

en_dict = defaultdict(int)
ch_dict = defaultdict(int)
total_dict = defaultdict(int)

with open(train_corpus, "r") as f:
    for i, line in enumerate(f):
        if "<en>" in line and "<ch>" in line:
            content = line.strip().split("<ch>")
            en_text = content[0].split()
            ch_text = content[1].split()
            for word in en_text:
                en_dict[word] += 1
                total_dict[word] += 1
            for word in ch_text:
                ch_dict[word] += 1
                total_dict[word] += 1
        elif "<en>" in line:
            content = line.strip().split()
            for word in content:
                en_dict[word] += 1
                total_dict[word] += 1
        elif "<ch>" in line:
            content = line.strip().split()
            for word in content:
                ch_dict[word] += 1
                total_dict[word] += 1
        else:
            print(line)



print("Chinese vocab size", len(ch_dict))  # origin: 3973   bpe: 3993
print("English vocab size", len(en_dict))  # origin: 27148  bpe: 20954
print("Total vocab size", len(total_dict)) # origin: 30753  bpe: 24538

sorted_total_dict = sorted(total_dict.items(), key=lambda item: item[1], reverse=True)

print(sorted_total_dict[:100])
with open(ench_dict_file, "w") as f:
    for word, cnt in sorted_total_dict:
        f.write(word + " " + str(cnt) + "\n")