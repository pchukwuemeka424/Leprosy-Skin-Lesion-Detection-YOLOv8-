#!/usr/bin/env python3
"""Rename all images that are not already l-<n>.<ext> to use l- numbering (l-40, l-41, ...)."""
import os
import re

IMGDIR = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".svg")

def main():
    os.chdir(IMGDIR)
    all_files = [
        f for f in os.listdir(".")
        if os.path.isfile(f) and not f.startswith(".")
        and os.path.splitext(f)[1].lower() in EXTENSIONS
    ]
    # Already l-<number>.<ext>?
    l_pattern = re.compile(r"^l-\d+\.[a-z]+$", re.I)
    need_rename = [f for f in sorted(all_files) if not l_pattern.match(f)]
    if not need_rename:
        print("All images already use l- naming.")
        return
    # Find max l- number
    max_n = 0
    for f in all_files:
        m = re.match(r"^l-(\d+)\.", f, re.I)
        if m:
            max_n = max(max_n, int(m.group(1)))
    next_n = max_n + 1
    renames = []
    for f in need_rename:
        ext = os.path.splitext(f)[1].lower()
        new = f"l-{next_n}{ext}"
        renames.append((f, new))
        next_n += 1
    # Rename via temp to avoid overwriting
    temp_renames = []
    for old, new in renames:
        temp = f"__tmp_{os.urandom(4).hex}_{old}"
        temp_renames.append((old, temp, new))
    for old, temp, new in temp_renames:
        os.rename(old, temp)
    for old, temp, new in temp_renames:
        os.rename(temp, new)
        print(f"{old} -> {new}")

if __name__ == "__main__":
    main()
