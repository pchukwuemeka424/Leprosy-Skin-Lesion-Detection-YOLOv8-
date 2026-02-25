#!/usr/bin/env python3
"""Rename images to existing convention: lowercase kebab-case (e.g. lepromatous-leprosy-face.png)."""
import os
import re

IMGDIR = os.path.dirname(os.path.abspath(__file__))

def to_kebab(s):
    s = s.replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
    s = re.sub(r"\.rf\.[a-f0-9]+", "", s)
    s = re.sub(r"-+", "-", s).strip("-").lower()
    return s

def main():
    os.chdir(IMGDIR)
    names = [
        f for f in os.listdir(".")
        if os.path.isfile(f) and "." in f and not f.startswith(".")
        and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")
    ]
    # Build old -> new mapping
    renames = {}
    image_jpeg_n = 1
    image_webp_n = 1
    for f in names:
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        if f == "images.jpeg":
            new = "image-1.jpeg"
        elif re.match(r"images \(\d+\)\.jpeg", f, re.I):
            n = int(re.search(r"\((\d+)\)", f).group(1))
            new = f"image-{n + 1}.jpeg"
        elif f == "image.webp":
            new = "image-1.webp"
        elif f == "image (1).webp":
            new = "image-2.webp"
        elif "Leprosy_thigh_demarcated" in f and "(1)" in f:
            new = "leprosy-thigh-demarcated-cutaneous-lesions-gallery4-2.jpg"
        elif "Leprosy_thigh_demarcated" in f:
            new = "leprosy-thigh-demarcated-cutaneous-lesions-gallery4.jpg"
        elif re.match(r"L-\d+-_jpg\.rf\.[a-f0-9]+\.jpg", f, re.I):
            num = re.search(r"L-(\d+)", f, re.I).group(1)
            new = f"l-{num}.jpg"
        elif f == "patch3.2e16d0ba.fill-430x430-c75.jpg":
            new = "patch-430x430.jpg"
        elif f == "cf809ce454f601c0cb724a18d4d6068a8afc5162.jpg":
            new = "figure-cf809ce.jpg"
        else:
            new_base = to_kebab(base)
            new = new_base + ext if new_base else f
        if new != f and new not in renames.values():
            renames[f] = new

    # Avoid overwriting: if target exists and is not in our source set, use temp
    targets = set(renames.values())
    for old, new in list(renames.items()):
        if new in targets and os.path.exists(new) and new != old:
            # target is another file we're renaming; will handle order
            pass
        elif os.path.exists(new) and new not in renames:
            # conflict with existing file
            base, ext = os.path.splitext(new)
            c = 1
            while os.path.exists(f"{base}-{c}{ext}"):
                c += 1
            renames[old] = f"{base}-{c}{ext}"

    # Rename via temp to avoid clashes
    temp_renames = []
    for old, new in renames.items():
        if old == new:
            continue
        temp = f"__tmp_{os.urandom(4).hex}_{old}"
        temp_renames.append((old, temp, new))
    for old, temp, new in temp_renames:
        os.rename(old, temp)
    for old, temp, new in temp_renames:
        os.rename(temp, new)
        print(f"{old} -> {new}")

if __name__ == "__main__":
    main()
