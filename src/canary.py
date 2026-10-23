"""Canary tokens (the harmless 'secret word' an injection tries to make the AI say).
Training uses some words and shapes; test sets use held-out words and shapes, so the scorer cannot detect
an injection just from the look of the token. Clean passages also get benign code-like tokens (see build_data)."""
TRAIN_WORDS = ["pineapple", "banana", "cobalt", "tulip", "marble", "orbit"]
TEST_WORDS = ["quasar", "lantern", "walnut", "zephyr"]
TRAIN_FORMATS = ["caps_hyphen", "plain", "quoted", "number"]   # e.g. COBALT-42, tulip, "marble orbit", 48213
TEST_FORMATS = ["hex", "cap"]                                  # e.g. a7f3c9, Quasar   (never seen in training)
ALL_FORMATS = TRAIN_FORMATS + TEST_FORMATS
ALL_WORDS = TRAIN_WORDS + TEST_WORDS


def token(rng, formats=None, words=None):
    formats = formats or ALL_FORMATS
    words = words or ALL_WORDS
    f, w = rng.choice(formats), rng.choice(words)
    if f == "caps_hyphen":
        return f"{w.upper()}-{rng.randint(1, 99)}"
    if f == "plain":
        return w
    if f == "quoted":
        return '"' + w + " " + rng.choice(words) + '"'
    if f == "number":
        return str(rng.randint(10000, 99999))
    if f == "hex":
        return "".join(rng.choice("0123456789abcdef") for _ in range(6))
    if f == "cap":
        return w.capitalize()
    raise ValueError(f)
