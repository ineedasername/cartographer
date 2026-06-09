"""ANSI color constants, terminal formatting, and CLI helpers."""

import sys
import unicodedata


# ─── ANSI color constants ─────────────────────────────────────────────────────

C_H   = '\033[95m'   # highlight (magenta)
C_B   = '\033[94m'   # blue
C_G   = '\033[92m'   # green
C_Y   = '\033[93m'   # yellow
C_C   = '\033[96m'   # cyan
C_E   = '\033[0m'    # reset
C_BLD = '\033[1m'    # bold
C_DIM = '\033[2m'    # dim


# ─── Terminal output helpers ──────────────────────────────────────────────────

def safe_print(text):
    """Print with fallback for Unicode encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def setup_stdout():
    """Configure stdout for UTF-8 on Windows. Call once at script startup."""
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ─── Heatmap coloring ────────────────────────────────────────────────────────

def _rgb(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'


HEAT = [
    _rgb(0xEC, 0x6A, 0x0F),  # < 0.4  — displaced (orange)
    _rgb(0xFF, 0xCF, 0x67),  # 0.4-0.75 — moderate (yellow)
    _rgb(0x21, 0x9C, 0x7F),  # >= 0.75 — stable (teal)
]


def heat(rho):
    """ANSI RGB color for a Spearman rho value. Three bands: displaced/moderate/stable."""
    if rho >= 0.75: return HEAT[2]
    if rho >= 0.40: return HEAT[1]
    return HEAT[0]


# ─── Token display formatting ─────────────────────────────────────────────────

def _char_width(ch):
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


def _display_width(s):
    return sum(_char_width(ch) for ch in s)


def fmt_tok(s, width=9):
    """Format token to exactly `width` terminal columns."""
    s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    s = s.lstrip('\u2581')
    s = s.lstrip(' ')
    if not s:
        s = '·'
    if _display_width(s) <= width:
        pad = width - _display_width(s)
        return s + ' ' * pad
    result = []
    used = 0
    for ch in s:
        ch_w = _char_width(ch)
        if ch_w > 0 and used + ch_w > width - 1:
            break
        result.append(ch)
        used += ch_w
    result.append('.')
    used += 1
    while used < width:
        result.append(' ')
        used += 1
    return ''.join(result)


def fmt_pct(p):
    """Format probability as two-digit percentage string."""
    pct = int(p * 100)
    if pct > 99:
        return "99"
    return f"{pct:02d}"


def prob_color(p):
    """ANSI color based on probability level."""
    if p > 0.5: return C_G
    if p > 0.1: return C_Y
    return C_DIM
