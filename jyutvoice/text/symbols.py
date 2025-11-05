from jyutvoice.text.cantonese.symbols import symbols as cantonese_symbols
from jyutvoice.text.english.symbols import symbols as english_symbols
from jyutvoice.text.mandarin.symbols import symbols as mandarin_symbols

punctuations = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = ["SP", "UNK"] + punctuations
pad = "_"

# Combine all unique symbols
all_symbols = list(set(cantonese_symbols + english_symbols + mandarin_symbols))
all_symbols.sort()  # Ensure consistent order

symbols = [pad] + pu_symbols + all_symbols
symbol_to_id = {s: i for i, s in enumerate(symbols)}

if __name__ == "__main__":
    print(f"Total combined symbols: {len(symbols)}")
    print("First 20 symbols:", symbols[:20])
    print("Cantonese symbols count:", len(cantonese_symbols))
    print("English symbols count:", len(english_symbols))
    print("Mandarin symbols count:", len(mandarin_symbols))
