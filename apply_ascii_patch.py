#!/usr/bin/env python3
"""
Re-apply the ASCII divider patch after re-downloading snapdragon_engine.py.

You patched _rule() locally so PowerShell renders dividers correctly. A fresh
download reverts that. Run this once after each re-download:

    python apply_ascii_patch.py

Safe to run twice -- it detects an already-patched file and does nothing.
"""
from pathlib import Path
import re, sys

p = Path("snapdragon_engine.py")
if not p.exists():
    sys.exit("snapdragon_engine.py not found; run this from the sigil folder.")

text = p.read_text(encoding="utf-8")
if 'return "-" * width' in text:
    print("Already ASCII-patched. Nothing to do.")
    sys.exit(0)

pattern = r'(?ms)^def _rule\(title: str = "", width: int = 78\) -> str:\r?\n.*?\r?\n\r?\n'
replacement = ('def _rule(title: str = "", width: int = 78) -> str:\n'
               '    if not title:\n'
               '        return "-" * width\n'
               '    pad = width - len(title) - 3\n'
               '    return f"-- {Colour.bold(title)} " + "-" * max(pad - 1, 0)\n\n\n')

m = re.search(pattern, text)
if m is None:
    sys.exit("ERROR: could not locate _rule(). No changes made.")

p.write_text(text[:m.start()] + replacement + text[m.end():],
             encoding="utf-8", newline="\n")
print("SUCCESS: dividers are ASCII. Re-run your commands.")
