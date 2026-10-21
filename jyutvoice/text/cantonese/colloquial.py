"""Colloquial reading corrections on top of ToJyutping's defaults (used by the JyutVoice Cantonese frontend and by scripts/synth_jyutvoice.py).

ToJyutping returns the literary/dictionary reading of several very common colloquial characters, and a single reading for polyphones whose reading
depends on context. This module corrects the cases below. Each fires only when the current reading is the default it was written to correct, and
`COLLOQUIAL_READINGS = False` turns everything off.

 1. TOKENS: word-level readings (HKCanCor, spoken Cantonese with manual Jyutping: >= 80% of >= 10 instances). Not included on purpose: 來 (always loi4;
    the colloquial "come" is written 嚟), context-dependent characters handled by the rules below, lazy-sound annotations, sentence-final particles.
 2. 無: the negator 冇 (mou5) unless it sits in a literary compound/idiom or a word of 3+ characters (names such as 無綫電視).
 3. 咁 before a punctuation mark or at the end of the text: gam2 (it is the manner/sentence-final 噉; written 咁 by most people).
 4. 假 after 放 請 病 年 ... : gaa3 ("holiday"); otherwise gaa2 ("fake").
 5. 正 as "great" (好正, 撚正): zeng3.
 6. 咪: mai5 ("don't") before an imperative continuation (咪傻, 咪話), otherwise mai6 ("then").
 7. 到 after 睇 見 聽 搵 ...: dou2 ("managed to see/find"), unless a degree/time word follows ("until"); 呢到 / 嗰到 (typos of 呢度 / 嗰度): dou6.
 8. CONTEXT_RULES: rules learned from HKCanCor for 下 到 返 樣 重 年 轉 位 使 坐 (previous/next character context, purity >= 90%,
    held-out accuracy of the rule set on HKCanCor utterances: 96.5% -> 97.8% of character positions, 98% of the overrides correct).
"""
import re

COLLOQUIAL_READINGS = True

TOKENS = {
    "返": ["faan1"], "即刻": ["zik1", "haak1"], "使": ["sai2"], "位": ["wai2"], "轉": ["zyun3"], "聲": ["seng1"],
    "哩啲": ["ni1", "di1"], "今年": ["gam1", "nin2"], "坐": ["co5"], "房": ["fong2"], "袋": ["doi2"], "魚": ["jyu2"], "相": ["soeng2"],
}
LITERARY_WU = {
    "無論", "無限", "無窮", "無奈", "無比", "無所", "無數", "無敵", "無辜", "無疑", "無可", "無情", "無形", "無聊", "無恥", "無知",
    "無端", "無謂", "無效", "無關", "無非", "無法", "無緣", "無需", "無故", "無私", "無暇", "無視", "無盡", "無名", "無畏", "無妨",
    "無礙", "無異", "無從", "無補", "無罪", "無價", "無憂", "無完", "無動", "無地", "無微", "無影", "無中", "無能", "無濟",
    "無綫", "無縫", "無蹤", "無品", "無家", "無反", "無憚", "無啦",
}
LITERARY_PREFIX = set("毫並絕了全別更體")
COLLOQUIAL_WU_NEXT = set("咁得咗嘢乜話錢事辦問嚟到晒有")
HOLIDAY_PREV = set("放請病年暑寒休事婚產例補度長短大日")
GREAT_PREV = set("好幾撚真咁超")
GREAT_NEXT_BLOCK = set("常式確義在面當")
IMPERATIVE_NEXT = set("話傻講理郁走動嘈吵再亂驚拎去做玩搞望睇聽急慌")
DOU2_PREV = set("睇見聽搵揾攞拎買收諗")
DOU3_NEXT = set("好幾咁半晚夜朝日最勁盡嘔甩累暈到唔想要成晒曬死")

# learned from HKCanCor: (kind, context, corrected reading, reading it corrects); kinds: c3 (prev, char, next), c2r (char, next), c2l (prev, char)
CONTEXT_RULES = [
    ('c2l', ('#', '使'), 'sai2', 'si2'),
    ('c2l', ('#', '重'), 'zung6', 'cung5'),
    ('c2l', ('今', '年'), 'nin2', 'nin4'),
    ('c2l', ('個', '樣'), 'joeng2', 'joeng6'),
    ('c2l', ('唔', '到'), 'dou2', 'dou3'),
    ('c2l', ('問', '下'), 'haa5', 'haa6'),
    ('c2l', ('揾', '到'), 'dou2', 'dou3'),
    ('c2l', ('攞', '到'), 'dou2', 'dou3'),
    ('c2l', ('收', '到'), 'dou2', 'dou3'),
    ('c2l', ('玩', '下'), 'haa5', 'haa6'),
    ('c2l', ('睇', '下'), 'haa5', 'haa6'),
    ('c2l', ('見', '到'), 'dou2', 'dou3'),
    ('c2l', ('試', '下'), 'haa5', 'haa6'),
    ('c2l', ('諗', '下'), 'haa5', 'haa6'),
    ('c2l', ('講', '下'), 'haa5', 'haa6'),
    ('c2l', ('講', '返'), 'faan1', 'faan2'),
    ('c2r', ('下', '你'), 'haa5', 'haa6'),
    ('c2r', ('下', '佢'), 'haa5', 'haa6'),
    ('c2r', ('下', '先'), 'haa5', 'haa6'),
    ('c2r', ('下', '啲'), 'haa5', 'haa6'),
    ('c2r', ('下', '噉'), 'haa5', 'haa6'),
    ('c2r', ('使', '唔'), 'sai2', 'si2'),
    ('c2r', ('坐', '喺'), 'co5', 'zo6'),
    ('c2r', ('轉', '#'), 'zyun3', 'zyun2'),
    ('c2r', ('返', '#'), 'faan1', 'faan2'),
    ('c2r', ('返', '佢'), 'faan1', 'faan2'),
    ('c2r', ('返', '出'), 'faan1', 'faan2'),
    ('c2r', ('返', '到'), 'faan1', 'faan2'),
    ('c2r', ('返', '啲'), 'faan1', 'faan2'),
    ('c2r', ('返', '屋'), 'faan1', 'faan2'),
    ('c2r', ('返', '返'), 'faan1', 'faan2'),
    ('c2r', ('重', '係'), 'zung6', 'cung5'),
    ('c2r', ('重', '好'), 'zung6', 'cung5'),
    ('c3', ('#', '返', '到'), 'faan1', 'faan2'),
    ('c3', ('#', '重', '係'), 'zung6', 'cung5'),
    ('c3', ('#', '重', '有'), 'zung6', 'cung5'),
    ('c3', ('#', '重', '要'), 'zung6', 'cung5'),
    ('c3', ('個', '樣', '#'), 'joeng2', 'joeng6'),
    ('c3', ('唔', '到', '#'), 'dou2', 'dou3'),
    ('c3', ('唔', '到', '啊'), 'dou2', 'dou3'),
    ('c3', ('唔', '到', '工'), 'dou2', 'dou3'),
    ('c3', ('問', '下', '你'), 'haa5', 'haa6'),
    ('c3', ('睇', '下', '#'), 'haa5', 'haa6'),
    ('c3', ('睇', '下', '你'), 'haa5', 'haa6'),
    ('c3', ('睇', '下', '先'), 'haa5', 'haa6'),
    ('c3', ('見', '到', '#'), 'dou2', 'dou3'),
    ('c3', ('見', '到', '佢'), 'dou2', 'dou3'),
    ('c3', ('見', '到', '有'), 'dou2', 'dou3'),
    ('c3', ('都', '重', '係'), 'zung6', 'cung5'),
]
_RULES = {(kind, key): (new, old) for kind, key, new, old in CONTEXT_RULES}
_HAN = re.compile(r"[㐀-䶿一-鿿]")


def _han(c):
    return c if c and _HAN.match(c) else "#"


def apply(words, per_word):
    """words: segmented words of one clause; per_word: [[syllable per character] per word] (ToJyutping readings).
    -> the same structure with colloquial readings."""
    out = [list(j) for j in per_word]
    if any(len(o) != len(w) for o, w in zip(out, words)):  # unexpected shape: leave ToJyutping's readings alone
        return per_word
    for k, w in enumerate(words):
        if w in TOKENS and len(TOKENS[w]) == len(w):
            out[k] = list(TOKENS[w])
    chars, flat = list("".join(words)), [x for o in out for x in o]
    tok = [w for w in words for _ in w]  # the segmented word each character belongs to
    default = [x for j in per_word for x in j]
    for i, ch in enumerate(chars):
        if flat[i] != default[i]:  # already changed by the word lexicon
            continue
        p, n = _han(chars[i - 1] if i else ""), _han(chars[i + 1] if i + 1 < len(chars) else "")
        cur = flat[i]
        if ch == "無" and cur == "mou4":
            if "".join(chars[i:i + 2]) in LITERARY_WU or p in LITERARY_PREFIX:
                continue
            w = tok[i]
            if len(w) >= 3 and not (w[0] == "無" and w[1] in COLLOQUIAL_WU_NEXT):
                continue
            flat[i] = "mou5"
        elif ch == "咁" and cur == "gam3" and n == "#":
            flat[i] = "gam2"
        elif ch == "假" and cur == "gaa2" and tok[i] == "假" and p in HOLIDAY_PREV:
            flat[i] = "gaa3"
        elif ch == "正" and cur == "zing3" and tok[i] == "正" and p in GREAT_PREV and n not in GREAT_NEXT_BLOCK:
            flat[i] = "zeng3"
        elif ch == "咪" and cur == "mai5" and tok[i] == "咪" and n not in IMPERATIVE_NEXT:
            flat[i] = "mai6"
        elif ch == "到" and cur == "dou3" and p in "呢嗰":  # typo for 度: 呢度 / 嗰度 (place words)
            flat[i] = "dou6"
        elif ch == "到" and cur == "dou3" and p in DOU2_PREV and n not in DOU3_NEXT:
            flat[i] = "dou2"
        else:
            for kind, key in (("c3", (p, ch, n)), ("c2r", (ch, n)), ("c2l", (p, ch))):
                r = _RULES.get((kind, key))
                if r and r[1] == cur:
                    flat[i] = r[0]
                    break
    res, k = [], 0
    for w in words:
        res.append(flat[k:k + len(w)])
        k += len(w)
    return res
