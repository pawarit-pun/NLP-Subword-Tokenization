"""Preprocess WikiText-103 for training BPE / WordPiece / Unigram tokenizers.

Run:  python preprocess/preprocess.py   (from the repo root)
In:   raw/wikitext-103-raw/wiki.{train,valid,test}.raw if present (original zip, no extra libraries),
      otherwise downloads from Hugging Face with `datasets`
Out:  data/train.txt, data/validation.txt, data/test.txt
      (UTF-8, one text per line, ready for tokenizer training)
"""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"
RAW_DIRS = [ROOT / "raw" / "wikitext-103-raw", ROOT / "raw"]
RAW_FILES = {"train": "wiki.train.raw", "validation": "wiki.valid.raw", "test": "wiki.test.raw"}

HEADING = re.compile(r"^(?:= )+(.+?)(?: =)+$")            # "= = Gameplay = =" -> "Gameplay"
CLITIC = re.compile(r" (n't|'(?:s|re|ve|d|ll|m|t))\b")    # "game 's" -> "game's"
QUOTE = re.compile(r'" (.*?) "')                           # '" word "' -> '"word"'
BEFORE_PUNCT = re.compile(r" ([,.;:!?%)\]}])")             # "word ," -> "word,"
AFTER_OPEN = re.compile(r"([(\[{]) ")                      # "( word" -> "(word"
SPACES = re.compile(r"\s+")


def clean(line: str) -> str | None:
    """Clean one line. Returns None if the line should be dropped."""
    text = line.strip()
    if not text:
        return None                                        # 1. drop empty lines

    m = HEADING.match(text)
    if m:
        text = m.group(1)                                  # 2. keep heading text, drop "=" markers

    text = unicodedata.normalize("NFC", text)              # 3. Unicode NFC

    text = (text.replace(" @-@ ", "-")                     # 4. WikiText markers back to normal
                .replace(" @,@ ", ",")                     #    "1 @,@ 000" -> "1,000"
                .replace(" @.@ ", "."))                    #    "3 @.@ 5"   -> "3.5"

    text = CLITIC.sub(r"\1", text)                         # 5. remove spaces WikiText put
    text = QUOTE.sub(r'"\1"', text)                        #    around punctuation
    text = BEFORE_PUNCT.sub(r"\1", text)
    text = AFTER_OPEN.sub(r"\1", text)

    text = SPACES.sub(" ", text).strip()                   # 6. single spaces only
    return text or None
    # Not done on purpose: lowercase, removing punctuation / numbers / stopwords.
    # The tokenizer has to learn those.


def load_splits():
    """Official train/validation/test split. Local raw files first, Hugging Face otherwise."""
    for d in RAW_DIRS:
        if all((d / f).exists() for f in RAW_FILES.values()):
            print(f"reading local files from {d}")
            return {split: (d / f).read_text(encoding="utf-8").splitlines()
                    for split, f in RAW_FILES.items()}
    from datasets import load_dataset
    print("downloading from Hugging Face")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return {split: ds[split]["text"] for split in RAW_FILES}


def main():
    raw = load_splits()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = {}
    for split in ("train", "validation", "test"):
        lines = [clean(t) for t in raw[split]]
        cleaned[split] = [t for t in lines if t is not None]
        print(f"{split:<10} {len(lines):>10,} raw lines -> {len(cleaned[split]):>10,} clean lines")

    # Keep test/validation unseen: drop long train lines that also appear there.
    held_out = {t for t in cleaned["validation"] + cleaned["test"] if len(t.split()) >= 5}
    before = len(cleaned["train"])
    cleaned["train"] = [t for t in cleaned["train"] if t not in held_out]
    print(f"removed {before - len(cleaned['train']):,} train lines that were also in validation/test")

    for split, lines in cleaned.items():
        path = OUT_DIR / f"{split}.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
