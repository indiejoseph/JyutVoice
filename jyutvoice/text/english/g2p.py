import pickle
import os
import re
from g2p_en import G2p
import wordsegment
from transformers import DebertaV2Tokenizer
from jyutvoice.text.symbols import punctuations
from builtins import str as unicode
from jyutvoice.text.english.symbols import symbols
from jyutvoice.text.english.normalization import normalize

from nltk.tokenize import TweetTokenizer

word_tokenize = TweetTokenizer().tokenize
from nltk import pos_tag

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
NAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")

LOCAL_PATH = "./bert/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                # 自定义发音词直接覆盖字典
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict


# 适配中文及 g2p_en 标点
rep_map = {
    "[;:：，；]": ",",
    '["’]': "'",
    "。": ".",
    "！": "!",
    "？": "?",
}


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def replace_consecutive_punctuation(text):
    puncts = "".join(re.escape(p) for p in punctuations)
    pattern = f"([{puncts}\s])([{puncts}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text):
    # todo: eng text normalize

    # 效果相同，和 chinese.py 保持一致
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    text = pattern.sub(lambda x: rep_map[x.group()], text)

    text = unicode(text)
    text = normalize(text)

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


class en_G2p(G2p):
    def __init__(self):
        super().__init__()
        # 分词初始化
        wordsegment.load()

        # 扩展过时字典, 添加姓名字典
        self.cmu = get_dict()
        self.namedict = get_namedict()

        # 剔除读音错误的几个缩写
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # 修正多音字
        self.homograph2features["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (
            ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            "JJ",
        )

    def __call__(self, text):
        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        word_prons = []
        for o_word, pos in tokens:
            # 还原 g2p_en 小写操作逻辑
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            # 先把单字母推出去
            elif len(word) == 1:
                # 单读 A 发音修正, 这里需要原格式 o_word 判断大写
                if o_word == "A":
                    pron = ["EY1"]
                else:
                    pron = self.cmu[word][0]
            # g2p_en 原版多音字处理
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                # pos1比pos长仅出现在read
                elif len(pos) < len(pos1) and pos == pos1[: len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                # 递归查找预测
                pron = self.qryword(o_word)

            word_prons.append((o_word, pron))

        return word_prons

    def qryword(self, o_word):
        word = o_word.lower()

        # 查字典, 单字母除外
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # 单词仅首字母大写时查找姓名字典
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # oov 长度小于等于 3 直接读字母
        if len(word) <= 3:
            phones = []
            for w in word:
                # 单读 A 发音修正, 此处不存在大写的情况
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # 尝试分离所有格
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH 无声辅音结尾 's 发 ['S']
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            # S Z SH ZH CH JH 擦声结尾 's 发 ['IH1', 'Z'] 或 ['AH0', 'Z']
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            # B D G DH V M N NG L R W Y 有声辅音结尾 's 发 ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IY0 IY1 IY2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 元音结尾 's 发 ['Z']
            else:
                phones.extend(["Z"])
            return phones

        # 尝试进行分词，应对复合词
        comps = wordsegment.segment(word.lower())

        # 无法分词的送回去预测
        if len(comps) == 1:
            return self.predict(word)

        # 可以分词的递归处理
        return [phone for comp in comps for phone in self.qryword(comp)]


_g2p = en_G2p()


def g2p(text, phoneme=None, padding=True):
    words = []
    phones = []
    tones = []
    word2ph = []
    syllable_pos = []
    word_pos = []
    ws_labels = []
    phone_len = []
    text = text_normalize(text)
    word_prons = _g2p(text)

    if phoneme is not None:
        raise NotImplementedError("Phoneme input is not supported yet.")

    for i, (w, pron) in enumerate(word_prons):
        temp_phones, temp_tones, temp_word2ph, temp_syllable_pos = [], [], [], []

        words.append(w)
        if w in punctuations:
            temp_phones.append(w)
            temp_tones.append(0)
            temp_word2ph.append(1)
            temp_syllable_pos.append(0)
        else:
            phns = []
            tns = []
            for ph in pron:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phns.append(ph)
                    tns.append(tn)
                else:
                    phns.append(ph)
                    tns.append(0)
            temp_phones += [post_replace_ph(i) for i in phns]
            temp_tones += tns
            temp_word2ph.append(len(phns))
            if len(pron) == 1 and pron[0] in punctuations:
                temp_syllable_pos = [0]
            else:
                for j in range(len(pron)):
                    if j == 0:
                        temp_syllable_pos.append(1)
                    elif j == len(pron) - 1:
                        temp_syllable_pos.append(3)
                    else:
                        temp_syllable_pos.append(2)

        if i != len(word_prons) - 1:
            temp_phones.append("_")  # word separator
            temp_tones.append(0)
            temp_word2ph.append(1)
            temp_syllable_pos.append(0)
            ws_labels.append(1)

        phones += temp_phones
        tones += temp_tones
        word2ph += temp_word2ph
        phone_len.append(len(temp_phones))
        ws_labels.append(1)  # English words are always single units
        syllable_pos += temp_syllable_pos

    word_pos = []
    for i, ws_label in enumerate(ws_labels):
        num_phones = word2ph[i]
        word_pos.extend([ws_label] * num_phones)

    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    if padding:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word_pos = [0] + word_pos + [0]
        syllable_pos = [0] + syllable_pos + [0]
        word2ph = [1] + word2ph + [1]

    lang_ids = [2] * len(phones)  # 2 for English

    return phones, tones, word2ph, word_pos, syllable_pos, lang_ids


if __name__ == "__main__":
    text = "In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."
    phones, tones, word2ph, word_pos, syllable_pos, lang_ids = g2p(text)
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
