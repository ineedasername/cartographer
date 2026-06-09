"""Rank-displacement scan summary and dashboard logic.

Aggregates per-position Spearman rho across scans and renders text summaries:
overall means, input-vs-generation breakdown, per-layer grids, and a
position-by-position evolution view.
"""

import sqlite3

from tokographer.common.cli import C_E


# ─── Heatmap coloring ────────────────────────────────────────────────────────

def _rgb(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

HEAT = [
    _rgb(0xEC, 0x6A, 0x0F),  # < 0.4  — displaced (orange)
    _rgb(0xFF, 0xCF, 0x67),  # 0.4-0.75 — moderate (yellow)
    _rgb(0x21, 0x9C, 0x7F),  # >= 0.75 — stable (teal)
]

def heat(rho):
    """Return ANSI color code for a Spearman rho value."""
    if rho >= 0.75: return HEAT[2]
    if rho >= 0.40: return HEAT[1]
    return HEAT[0]


# ─── Summary functions ───────────────────────────────────────────────────────

def get_scan_summary(conn):
    """Get overall scan summary rows.

    Args:
        conn: sqlite3.Connection with row_factory=sqlite3.Row

    Returns:
        list of dicts with scan_id, prompt, n_input_tokens, n_gen_tokens, avg_spearman, min_spearman
    """
    scans = conn.execute('SELECT * FROM scans ORDER BY scan_id').fetchall()
    results = []
    for s in scans:
        row = conn.execute('SELECT AVG(spearman) as a, MIN(spearman) as m FROM positions WHERE scan_id=?',
                           (s['scan_id'],)).fetchone()
        results.append({
            'scan_id': s['scan_id'],
            'prompt': s['prompt'],
            'n_input_tokens': s['n_input_tokens'],
            'n_gen_tokens': s['n_gen_tokens'],
            'avg_spearman': row['a'] or 0,
            'min_spearman': row['m'] or 0,
        })
    return results


def get_phase_summary(conn):
    """Get mean Spearman by phase (input vs gen) per scan.

    Returns:
        list of dicts with scan_id, prompt, input_rho, gen_rho, delta
    """
    scans = conn.execute('SELECT * FROM scans ORDER BY scan_id').fetchall()
    results = []
    for s in scans:
        inp = conn.execute("SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND phase='input'",
                           (s['scan_id'],)).fetchone()['a'] or 0
        gen = conn.execute("SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND phase='gen'",
                           (s['scan_id'],)).fetchone()['a'] or 0
        results.append({
            'scan_id': s['scan_id'],
            'prompt': s['prompt'],
            'input_rho': inp,
            'gen_rho': gen,
            'delta': gen - inp,
        })
    return results


def get_layer_summary(conn, phase='input'):
    """Get mean Spearman by layer for each scan.

    Args:
        conn: sqlite3.Connection
        phase: 'input' or 'gen'

    Returns:
        (layers, scans, grid) where grid[layer_idx][scan_idx] = mean rho
    """
    scans = conn.execute('SELECT * FROM scans ORDER BY scan_id').fetchall()
    layers = sorted(set(r['layer'] for r in conn.execute(
        "SELECT DISTINCT layer FROM positions").fetchall()))

    grid = []
    for l in layers:
        row = []
        for s in scans:
            r = conn.execute(
                "SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND layer=? AND phase=?",
                (s['scan_id'], l, phase)).fetchone()
            row.append(r['a'] if r['a'] else 0)
        grid.append(row)

    return layers, scans, grid


def print_dashboard(db_path):
    """Print the full rank-displacement dashboard to stdout.

    Args:
        db_path: str path to SQLite DB
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    scans = conn.execute('SELECT * FROM scans ORDER BY scan_id').fetchall()
    if not scans:
        print("No scans found.")
        conn.close()
        return

    # Overall summary
    print(f"\n{'='*75}")
    print(f"  SCAN SUMMARY")
    print(f"{'='*75}")
    print(f"  {'ID':>3} {'Prompt':<40} {'In':>4} {'Gen':>4} {'Mean p':>7} {'Min p':>7}")
    print(f"  {'-'*68}")
    for s in scans:
        row = conn.execute('SELECT AVG(spearman) as a, MIN(spearman) as m FROM positions WHERE scan_id=?',
                           (s['scan_id'],)).fetchone()
        prompt_short = repr(s['prompt'])[:38]
        print(f"  {s['scan_id']:>3} {prompt_short:<40} {s['n_input_tokens']:>4} {s['n_gen_tokens']:>4} "
              f"{row['a']:>7.4f} {row['m']:>7.4f}")

    # By phase
    print(f"\n{'='*75}")
    print(f"  MEAN SPEARMAN BY PHASE")
    print(f"{'='*75}")
    print(f"  {'Scan':>4} {'Prompt':<30} {'Input p':>8} {'Gen p':>8} {'D':>8}")
    print(f"  {'-'*60}")
    for s in scans:
        inp = conn.execute("SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND phase='input'",
                           (s['scan_id'],)).fetchone()['a'] or 0
        gen = conn.execute("SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND phase='gen'",
                           (s['scan_id'],)).fetchone()['a'] or 0
        delta = gen - inp
        print(f"  {s['scan_id']:>4} {repr(s['prompt'])[:28]:<30} {inp:>8.4f} {gen:>8.4f} {delta:>+8.4f}")

    # By layer (both phases)
    layers = sorted(set(r['layer'] for r in conn.execute(
        "SELECT DISTINCT layer FROM positions").fetchall()))

    for phase in ('input', 'gen'):
        print(f"\n{'='*75}")
        print(f"  MEAN SPEARMAN BY LAYER ({phase} phase)")
        print(f"{'='*75}")
        print(f"  {'Layer':>5}", end='')
        for s in scans:
            label = repr(s['prompt'])[:10]
            print(f"  {label:>10}", end='')
        print()

        for l in layers:
            print(f"  L{l:02d}  ", end='')
            for s in scans:
                row = conn.execute(
                    "SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND layer=? AND phase=?",
                    (s['scan_id'], l, phase)).fetchone()
                val = row['a'] if row['a'] else 0
                c = heat(val)
                print(f"  {c}{val:>10.4f}{C_E}", end='')
            print()

    # Position evolution
    print(f"\n{'='*75}")
    print(f"  SPEARMAN EVOLUTION BY INPUT POSITION (mean across all LxH)")
    print(f"{'='*75}")
    print(f"  {'Pos':>4} {'Token':>15}", end='')
    for s in scans:
        label = 'S' + str(s['scan_id'])
        print(f"  {label:>8}", end='')
    print()

    max_pos = max(s['n_input_tokens'] for s in scans) + max(s['n_gen_tokens'] for s in scans)
    for pos in range(max_pos):
        tok_text = ''
        phase = 'input'
        for s in scans:
            row = conn.execute("SELECT token_text, phase FROM positions WHERE scan_id=? AND position=? LIMIT 1",
                               (s['scan_id'], pos)).fetchone()
            if row:
                tok_text = row['token_text']
                phase = row['phase']
                break
        if not tok_text and pos > 0:
            continue

        marker = '|' if phase == 'gen' else ' '
        print(f"  {pos:>3}{marker} {repr(tok_text)[:13]:>15}", end='')
        for s in scans:
            row = conn.execute("SELECT AVG(spearman) as a FROM positions WHERE scan_id=? AND position=?",
                               (s['scan_id'], pos)).fetchone()
            if row and row['a'] is not None:
                c = heat(row['a'])
                print(f"  {c}{row['a']:>8.4f}{C_E}", end='')
            else:
                print(f"  {'':>8}", end='')
        print()

    conn.close()
