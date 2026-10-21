#!/usr/bin/env python3
"""Make Cantonese text safe for the fine-tuned CosyVoice2 teacher (and for the student trained on its output).

The teacher's training texts contain only the punctuation `，。？！`, no digits, almost no Latin letters, and only
Traditional characters, so anything else makes it skip, garble or invent sounds. `CantoneseTextFilter` turns a raw piece of
text into text the teacher can read, or rejects it with a reason:

  markup      `<...>`, backticks, brackets, backslashes (leaked reasoning tags etc.)
  latin       any Latin letter (code-mixed English-Cantonese reads badly)
  currency    $ / yen / euro ... (HK$ vs US$ is not something to guess)
  time        clock times like 10:30
  number      a number format we do not convert with confidence (dates like 2024-06-07, 8+ digit ids/phone numbers)
  digits_left digits or Latin still present after normalisation
  symbol      a character other than CJK and `，。？！` is left after mapping
  rare_char   a character seen fewer than --min_char_count times in the teacher's training texts (covers Simplified
              leakage and rare hanzi)
  short       fewer than 5 Chinese characters

Numbers are converted with a small conservative normaliser (cn2an for cardinals): years digit by digit, percentages,
decimals, fractions, ranges, negatives, plain integers. The general-purpose `tn` normaliser was rejected: it spells
digits one by one when a Traditional measure word follows (200個 -> 二零零個) and, with CosyVoice's default call, converts
to Simplified, deletes particles like 啊 and makes punctuation ASCII. Simplified numeral glyphs from cn2an are mapped back
(万亿两点负 -> 萬億兩點負), and 二 becomes 兩 before classifiers and 百千萬億 (2個 -> 兩個, 200蚊 -> 兩百蚊).

Punctuation: 「」『』《》〈〉“”‘’（）【】 and the middle dots are removed; 、；：dashes and ellipses become `，`; pieces end with 。.

    python text_filter.py --build_char_counts dataset20250907.csv --out train_char_counts.json
    python text_filter.py --selftest [--char_counts train_char_counts.json]
"""
import argparse
import collections
import csv
import json
import re
import sys

CJK = re.compile(r"[㐀-䶿一-鿿]")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
DIGIT = re.compile(r"[0-9０-９]")
MARKUP = re.compile(r"[<>`\\{}\[\]|]")
CURRENCY = re.compile(r"[$＄¥￥€£₩]")
TIME = re.compile(r"\d\s*[:：]\s*\d")
SUPPORTED = "，。？！"

FULLWIDTH = str.maketrans({"％": "%", "．": ".", "／": "/", "－": "-", **{chr(0xFF10 + i): str(i) for i in range(10)}})
REMOVE = str.maketrans("", "", "「」『』《》〈〉“”‘’\"'（）()【】［］·‧・")
TO_COMMA = str.maketrans({"、": "，", "；": "，", "：": "，", "—": "，", "–": "，", "―": "，", "─": "，", "…": "，", "⋯": "，"})
POST_NUMERAL = str.maketrans({"万": "萬", "亿": "億", "两": "兩", "点": "點", "负": "負", "幺": "一"})
ASCII_PUNCT = str.maketrans({",": "，", ";": "，", ":": "，", "!": "！", "?": "？", ".": "。"})
DIGITWISE = dict(zip("0123456789", "零一二三四五六七八九"))

CLASSIFIERS = "個隻位名條間次件張架部本杯碗種樣支對份所座幢"
NUMERAL_CHARS = "零一二三四五六七八九十百千萬億點第兩"
ER_BEFORE_CLASSIFIER = re.compile(f"(?<![{NUMERAL_CHARS}])二(?=[{CLASSIFIERS}])")
ER_BEFORE_UNIT = re.compile(f"(?<![{NUMERAL_CHARS}])二(?=[百千萬億])")


class NumberError(Exception):
    pass


def _cardinal(s):
    from cn2an import an2cn

    if len(s) > 1 and s.startswith("0"):  # 003 -> digit by digit
        return "".join(DIGITWISE[c] for c in s)
    if len(s) >= 13:
        raise NumberError(s)
    return an2cn(s, "low")


def _decimal(s):
    if "." in s:
        a, b = s.split(".", 1)
        return _cardinal(a) + "點" + "".join(DIGITWISE[c] for c in b)
    return _cardinal(s)


def normalize_numbers(t):
    """Digits -> Chinese numerals (Simplified glyphs for 万亿两, fixed later by POST_NUMERAL). Raises NumberError."""
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    if re.search(r"\d+[./-]\d+[./-]\d+", t) or re.search(r"\d{8,}", t):
        raise NumberError("date/id/phone")
    t = re.sub(r"(?<!\d)-(\d+(?:\.\d+)?)%", lambda m: "百分之負" + _decimal(m.group(1)), t)
    t = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: "百分之" + _decimal(m.group(1)), t)
    t = re.sub(r"(\d+)/(\d+)", lambda m: _cardinal(m.group(2)) + "分之" + _cardinal(m.group(1)), t)
    def _range(m):  # 10-20 is a range; 6-0 (first >= second) is most likely a sports score, so give up on it
        a, b = m.group(1), m.group(2)
        if float(a) >= float(b):
            raise NumberError("score")
        return a + "至" + b

    t = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?![\d.])", _range, t)
    t = re.sub(r"(?<!\d)-(?=\d)", "負", t)
    # years are read digit by digit, including the first year of a range (1973至1980年)
    t = re.sub(r"(?<!\d)(\d{4})(?=至\d{4}年)", lambda m: "".join(DIGITWISE[c] for c in m.group(1)), t)
    t = re.sub(r"(?<!\d)(\d{4})(?=年)", lambda m: "".join(DIGITWISE[c] for c in m.group(1)), t)
    t = re.sub(r"\d+\.\d+", lambda m: _decimal(m.group(0)), t)
    t = re.sub(r"\d+", lambda m: _cardinal(m.group(0)), t)
    return t


def has_markup(text):
    return bool(MARKUP.search(text))


def _collapse(m):
    s = m.group(0)
    for p in "？！。":
        if p in s:
            return p
    return "，"


class CantoneseTextFilter:
    def __init__(self, digits="normalize", char_counts=None, min_char_count=20):
        assert digits in ("normalize", "drop")
        self.digits = digits
        self.min_char_count = min_char_count
        self.char_counts = None
        if char_counts:
            self.char_counts = json.load(open(char_counts, encoding="utf-8")) if isinstance(char_counts, str) else char_counts
        self.stats = collections.Counter()

    def __call__(self, raw):
        """Return (clean_text, reason). clean_text is None when rejected; reason then says why."""
        text, reason = self._run(raw)
        self.stats[reason or "kept"] += 1
        if text and DIGIT.search(raw):
            self.stats["kept_with_numbers"] += 1
        return text, reason

    def _run(self, raw):
        t = raw
        if has_markup(t):
            return None, "markup"
        if LATIN.search(t):
            return None, "latin"
        if CURRENCY.search(t):
            return None, "currency"
        if TIME.search(t):
            return None, "time"
        t = re.sub(r"\s+", "", t).translate(FULLWIDTH)
        t = t.translate(REMOVE).translate(TO_COMMA)
        if DIGIT.search(t):
            if self.digits == "drop":
                return None, "digits"
            try:
                t = normalize_numbers(t)
            except NumberError:
                return None, "number"
        t = t.translate(POST_NUMERAL)
        t = ER_BEFORE_CLASSIFIER.sub("兩", t)
        t = ER_BEFORE_UNIT.sub("兩", t)
        t = t.translate(ASCII_PUNCT)
        if DIGIT.search(t) or LATIN.search(t):
            return None, "digits_left"
        if any((not CJK.match(c)) and c not in SUPPORTED for c in t):
            return None, "symbol"
        t = re.sub(f"[{SUPPORTED}]{{2,}}", _collapse, t).lstrip(SUPPORTED)
        if t.endswith("，"):
            t = t[:-1] + "。"
        elif t and t[-1] not in "。？！":
            t += "。"
        if len(CJK.findall(t)) < 5:
            return None, "short"
        if self.char_counts is not None:
            if any(self.char_counts.get(c, 0) < self.min_char_count for c in set(CJK.findall(t))):
                return None, "rare_char"
        return t, None


def build_char_counts(csv_path, out_path, limit=None):
    csv.field_size_limit(10**9)
    counts = collections.Counter()
    n = 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts.update(CJK.findall(row["text"]))
            n += 1
            if limit and n >= limit:
                break
    json.dump(counts, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{n:,} texts, {len(counts):,} distinct characters -> {out_path}")


CASES = [
    # (input, expected output or None, expected reject reason)
    ("2024年6月7日，恒指升3.5%，成交額超過1,234億元。", "二零二四年六月七日，恒指升百分之三點五，成交額超過一千二百三十四億元。", None),
    ("佢哋買咗2個，真係好嘢啊！", "佢哋買咗兩個，真係好嘢啊！", None),
    ("呢個係第2間店，亦都有12個人。", "呢個係第二間店，亦都有十二個人。", None),
    ("約有2萬人參加，要俾200蚊。", "約有兩萬人參加，要俾兩百蚊。", None),
    ("我今年25歲，已經工作咗10年。", "我今年二十五歲，已經工作咗十年。", None),
    ("廖博士喺1973至1980年間擔任署長，之後1980-1985年轉職。", "廖博士喺一九七三至一九八零年間擔任署長，之後一九八零至一九八五年轉職。", None),
    ("佢升咗5%同埋跌咗3/4。", "佢升咗百分之五同埋跌咗四分之三。", None),
    ("股價跌咗-5%，由10至20元，再跌10-20蚊。", "股價跌咗百分之負五，由十至二十元，再跌十至二十蚊。", None),
    ("圓周率大約係3.14倍。", "圓周率大約係三點一四倍。", None),
    ("佢住003號房，今年係2025年。", "佢住零零三號房，今年係二零二五年。", None),
    ("我今年25歲，電話係12345678。", None, "number"),
    ("事件發生喺2024-06-07。", None, "number"),
    ("曼聯喺上半場以6-0領先對手。", None, "number"),
    ("會議10:30開始，大家要準時。", None, "time"),
    ("佢哋講「早晨」，仲睇咗《無間道》。", "佢哋講早晨，仲睇咗無間道。", None),
    ("佢哋、我哋；同你哋：全部都嚟咗", "佢哋，我哋，同你哋，全部都嚟咗。", None),
    ("佢（我嘅朋友）琴日嚟過。", "佢我嘅朋友琴日嚟過。", None),
    ("<chain_of_thoughts>先分析一下</chain_of_thoughts>今日天氣好好。", None, "markup"),
    ("</think>今日天氣好好。", None, "markup"),
    ("我哋用咗Google搜尋。", None, "latin"),
    ("成交價係HK$500。", None, "latin"),
    ("成交價係$500。", None, "currency"),
    ("佢哋之間有~好大差距。", None, "symbol"),
    ("今日天氣好熱。", "今日天氣好熱。", None),
    ("好。", None, "short"),
]


def selftest(char_counts=None):
    flt = CantoneseTextFilter()
    bad = 0
    for raw, want, why in CASES:
        got, reason = flt(raw)
        ok = (got == want) and (reason == why)
        bad += not ok
        print(("ok   " if ok else "FAIL ") + f"{raw[:34]!r:38s} -> {got!r} {reason or ''}" + ("" if ok else f"   (wanted {want!r} {why or ''})"))
    if char_counts:
        flt2 = CantoneseTextFilter(char_counts=char_counts, min_char_count=20)
        for raw, want_reason in [("呢個係测试嘅句子。", "rare_char"), ("佢今日好開心。", None)]:
            got, reason = flt2(raw)
            ok = reason == want_reason
            bad += not ok
            print(("ok   " if ok else "FAIL ") + f"{raw!r:20s} -> {got!r} {reason or ''}")
    print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_char_counts")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--char_counts")
    a = ap.parse_args()
    if a.build_char_counts:
        build_char_counts(a.build_char_counts, a.out, a.limit)
    if a.selftest:
        sys.exit(1 if selftest(a.char_counts) else 0)
