"""Token categorization, Unicode script detection, and display utilities.

Lifted from:
  - telescope_isa.py L42-119 (categorize_token, broader_category)
  - build_token_index.py L40-165 (_SCRIPT_RANGES, char_script)
  - projection_map.py L35-49 (char_width, display_width)
"""

import unicodedata


# ─── Unicode script detection ─────────────────────────────────────────────────
# Source: build_token_index.py L40-165

_SCRIPT_RANGES = [
    (0x0000, 0x007F, 'Latin'),       # Basic Latin (ASCII)
    (0x0080, 0x00FF, 'Latin'),       # Latin-1 Supplement
    (0x0100, 0x024F, 'Latin'),       # Latin Extended-A/B
    (0x0250, 0x02AF, 'Latin'),       # IPA Extensions
    (0x0300, 0x036F, 'Combining'),   # Combining Diacritical Marks
    (0x0370, 0x03FF, 'Greek'),
    (0x0400, 0x04FF, 'Cyrillic'),
    (0x0500, 0x052F, 'Cyrillic'),    # Cyrillic Supplement
    (0x0530, 0x058F, 'Armenian'),
    (0x0590, 0x05FF, 'Hebrew'),
    (0x0600, 0x06FF, 'Arabic'),
    (0x0700, 0x074F, 'Syriac'),
    (0x0750, 0x077F, 'Arabic'),      # Arabic Supplement
    (0x0780, 0x07BF, 'Thaana'),
    (0x07C0, 0x07FF, 'NKo'),
    (0x0800, 0x083F, 'Samaritan'),
    (0x0900, 0x097F, 'Devanagari'),
    (0x0980, 0x09FF, 'Bengali'),
    (0x0A00, 0x0A7F, 'Gurmukhi'),
    (0x0A80, 0x0AFF, 'Gujarati'),
    (0x0B00, 0x0B7F, 'Oriya'),
    (0x0B80, 0x0BFF, 'Tamil'),
    (0x0C00, 0x0C7F, 'Telugu'),
    (0x0C80, 0x0CFF, 'Kannada'),
    (0x0D00, 0x0D7F, 'Malayalam'),
    (0x0D80, 0x0DFF, 'Sinhala'),
    (0x0E00, 0x0E7F, 'Thai'),
    (0x0E80, 0x0EFF, 'Lao'),
    (0x0F00, 0x0FFF, 'Tibetan'),
    (0x1000, 0x109F, 'Myanmar'),
    (0x10A0, 0x10FF, 'Georgian'),
    (0x1100, 0x11FF, 'Hangul'),      # Hangul Jamo
    (0x1200, 0x137F, 'Ethiopic'),
    (0x1380, 0x139F, 'Ethiopic'),    # Ethiopic Supplement
    (0x13A0, 0x13FF, 'Cherokee'),
    (0x1400, 0x167F, 'Canadian'),    # Unified Canadian Aboriginal Syllabics
    (0x1680, 0x169F, 'Ogham'),
    (0x16A0, 0x16FF, 'Runic'),
    (0x1700, 0x171F, 'Tagalog'),
    (0x1720, 0x173F, 'Hanunoo'),
    (0x1780, 0x17FF, 'Khmer'),
    (0x1800, 0x18AF, 'Mongolian'),
    (0x1900, 0x194F, 'Limbu'),
    (0x1E00, 0x1EFF, 'Latin'),       # Latin Extended Additional
    (0x1F00, 0x1FFF, 'Greek'),       # Greek Extended
    (0x2000, 0x206F, 'Punctuation'), # General Punctuation
    (0x2070, 0x209F, 'Latin'),       # Superscripts/Subscripts
    (0x20A0, 0x20CF, 'Symbol'),      # Currency Symbols
    (0x2100, 0x214F, 'Symbol'),      # Letterlike Symbols
    (0x2150, 0x218F, 'Symbol'),      # Number Forms
    (0x2190, 0x21FF, 'Symbol'),      # Arrows
    (0x2200, 0x22FF, 'Symbol'),      # Mathematical Operators
    (0x2300, 0x23FF, 'Symbol'),      # Miscellaneous Technical
    (0x2500, 0x257F, 'Symbol'),      # Box Drawing
    (0x2580, 0x259F, 'Symbol'),      # Block Elements
    (0x25A0, 0x25FF, 'Symbol'),      # Geometric Shapes
    (0x2600, 0x26FF, 'Symbol'),      # Miscellaneous Symbols
    (0x2700, 0x27BF, 'Symbol'),      # Dingbats
    (0x2E80, 0x2EFF, 'Han'),         # CJK Radicals Supplement
    (0x2F00, 0x2FDF, 'Han'),         # Kangxi Radicals
    (0x3000, 0x303F, 'CJK_Punct'),   # CJK Symbols and Punctuation
    (0x3040, 0x309F, 'Hiragana'),
    (0x30A0, 0x30FF, 'Katakana'),
    (0x3100, 0x312F, 'Bopomofo'),
    (0x3130, 0x318F, 'Hangul'),      # Hangul Compatibility Jamo
    (0x3190, 0x319F, 'Han'),         # Kanbun
    (0x31A0, 0x31BF, 'Bopomofo'),
    (0x31F0, 0x31FF, 'Katakana'),    # Katakana Phonetic Extensions
    (0x3200, 0x32FF, 'Han'),         # Enclosed CJK Letters
    (0x3300, 0x33FF, 'Han'),         # CJK Compatibility
    (0x3400, 0x4DBF, 'Han'),         # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF, 'Han'),         # CJK Unified Ideographs
    (0xA000, 0xA48F, 'Yi'),
    (0xA490, 0xA4CF, 'Yi'),
    (0xAC00, 0xD7AF, 'Hangul'),      # Hangul Syllables
    (0xD7B0, 0xD7FF, 'Hangul'),      # Hangul Jamo Extended-B
    (0xF900, 0xFAFF, 'Han'),         # CJK Compatibility Ideographs
    (0xFB00, 0xFB06, 'Latin'),       # Alphabetic Presentation Forms
    (0xFB1D, 0xFB4F, 'Hebrew'),      # Hebrew Presentation Forms
    (0xFB50, 0xFDFF, 'Arabic'),      # Arabic Presentation Forms-A
    (0xFE30, 0xFE4F, 'Han'),         # CJK Compatibility Forms
    (0xFE70, 0xFEFF, 'Arabic'),      # Arabic Presentation Forms-B
    (0xFF00, 0xFFEF, 'Fullwidth'),   # Halfwidth and Fullwidth Forms
    (0x10000, 0x1007F, 'Linear_B'),
    (0x10080, 0x100FF, 'Linear_B'),
    (0x10300, 0x1032F, 'Old_Italic'),
    (0x10330, 0x1034F, 'Gothic'),
    (0x10400, 0x1044F, 'Deseret'),
    (0x1D000, 0x1D0FF, 'Symbol'),    # Byzantine Musical Symbols
    (0x1D100, 0x1D1FF, 'Symbol'),    # Musical Symbols
    (0x1D400, 0x1D7FF, 'Symbol'),    # Mathematical Alphanumeric Symbols
    (0x1F000, 0x1F02F, 'Symbol'),    # Mahjong Tiles
    (0x1F030, 0x1F09F, 'Symbol'),    # Domino Tiles
    (0x1F100, 0x1F1FF, 'Symbol'),    # Enclosed Alphanumeric Supplement
    (0x1F200, 0x1F2FF, 'Han'),       # Enclosed Ideographic Supplement
    (0x1F300, 0x1F9FF, 'Emoji'),     # Miscellaneous Symbols / Emoticons / etc
    (0x1FA00, 0x1FA6F, 'Emoji'),
    (0x1FA70, 0x1FAFF, 'Emoji'),
    (0x20000, 0x2A6DF, 'Han'),       # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F, 'Han'),       # CJK Extension C
    (0x2B740, 0x2B81F, 'Han'),       # CJK Extension D
    (0x2B820, 0x2CEAF, 'Han'),       # CJK Extension E
    (0x2CEB0, 0x2EBEF, 'Han'),       # CJK Extension F
    (0x30000, 0x3134F, 'Han'),       # CJK Extension G
]

_SCRIPT_RANGES.sort(key=lambda x: x[0])


def char_script(ch):
    """Detect the Unicode script of a single character via binary search."""
    cp = ord(ch)
    lo, hi = 0, len(_SCRIPT_RANGES) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, script = _SCRIPT_RANGES[mid]
        if cp < start:
            hi = mid - 1
        elif cp > end:
            lo = mid + 1
        else:
            return script
    return 'Unknown'


# ─── Terminal width calculation ───────────────────────────────────────────────
# Source: projection_map.py L35-49

def char_width(ch):
    """Terminal column width of a single character."""
    cat = unicodedata.category(ch)
    if cat in ('Mn', 'Me'):
        return 0
    if ord(ch) in (0x200B, 0x200C, 0x200D, 0xFEFF):
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ('W', 'F'):
        return 2
    return 1


def display_width(s):
    """Terminal column width of a string."""
    return sum(char_width(ch) for ch in s)


# ─── Token categorization ────────────────────────────────────────────────────
# Source: telescope_isa.py L42-119

def categorize_token(token_str, token_id):
    """Classify an output token into a semantic category.

    Returns one of 13 fine categories:
        empty, special, digit, bracket, punct, quote, newline, symbol,
        code_keyword, code_identifier, foreign, function_word, short_word,
        content_word, mixed
    """
    t = token_str.strip()

    if not t or token_id == 0:
        return 'empty'

    if t.startswith('<') and t.endswith('>'):
        return 'special'

    if t.isdigit():
        return 'digit'

    if all(not c.isalnum() for c in t):
        if t in ('(', ')', '[', ']', '{', '}'):
            return 'bracket'
        if t in ('.', ',', '!', '?', ':', ';'):
            return 'punct'
        if t in ("'", '"', '`', '```'):
            return 'quote'
        if t in ('\n', '\n\n', '\r\n'):
            return 'newline'
        return 'symbol'

    if t in ('def', 'if', 'else', 'for', 'while', 'return', 'import', 'class',
             'True', 'False', 'None', 'self', 'int', 'str', 'float', 'len'):
        return 'code_keyword'
    if '_' in t and t.replace('_', '').isalnum():
        return 'code_identifier'

    if any(ord(c) > 127 for c in t):
        return 'foreign'

    function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
                       'and', 'or', 'but', 'not', 'it', 'its', 'that', 'this',
                       'I', 'you', 'he', 'she', 'I', 'they', 'my', 'your', 'his',
                       'her', 'our', 'their', 'me', 'him', 'us', 'them',
                       's', 't', 're', 've', 'll', 'd', 'm'}
    if t.lower() in function_words:
        return 'function_word'

    if t.isalpha() or (t[0].isalpha() and t.replace("'", "").isalpha()):
        if len(t) <= 3:
            return 'short_word'
        return 'content_word'

    return 'mixed'


def broader_category(cat):
    """Collapse fine token categories to 6 broad categories.

    Returns one of: numeric, code, structure, language, foreign, other
    """
    mapping = {
        'digit': 'numeric',
        'code_keyword': 'code',
        'code_identifier': 'code',
        'bracket': 'structure',
        'punct': 'structure',
        'quote': 'structure',
        'newline': 'structure',
        'symbol': 'structure',
        'special': 'structure',
        'empty': 'structure',
        'foreign': 'foreign',
        'function_word': 'language',
        'short_word': 'language',
        'content_word': 'language',
        'mixed': 'language',
    }
    return mapping.get(cat, 'other')
