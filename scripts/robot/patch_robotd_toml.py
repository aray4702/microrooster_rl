#!/usr/bin/env python3
"""Set robotd.toml keys idempotently, printing what changed. Runs ON the robot.

A sed-per-key does the wrong thing when a key is absent (silently no-ops) or
commented out (edits the comment), and robotd.toml ships with most values
commented. This sets a key whether it is present, commented, or missing, and
prints a before/after so the change is visible in the terminal that made it.

    sudo python3 patch_robotd_toml.py walk=hopfree.onnx gain=400 ...
    sudo python3 patch_robotd_toml.py sitstand=!unset

`key=!unset` COMMENTS THE KEY OUT rather than deleting it, which is what the
hop/walk switch needs to be a true inverse: the hop setup sets
`sitstand = "none"` and normal operation has no such line at all (verified
against the pre-boot backup). Commenting rather than deleting keeps the file
self-documenting, and the "revive a commented default" path above brings it
straight back when the key is next set.
"""
import re
import shutil
import sys
import time

PATH = "/etc/robot/robotd.toml"


def main(argv: list[str]) -> int:
    pairs = []
    for arg in argv:
        if "=" not in arg:
            print(f"skipping {arg!r}: expected key=value")
            return 2
        k, v = arg.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    if not pairs:
        print("nothing to do")
        return 0

    original = open(PATH).read()
    backup = f"{PATH}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(PATH, backup)
    text = original

    for key, value in pairs:
        if value == "!unset":
            active = re.compile(rf"^(\s*)({re.escape(key)}\s*=.*)$", re.M)
            m = active.search(text)
            if m:
                text = active.sub(r"\1# \2", text, count=1)
                print(f"  {key:16s} {m.group(0).strip()}  ->  # (unset)")
            else:
                print(f"  {key:16s} <absent or already unset>  ->  # (unset)")
            continue
        # An active assignment wins; otherwise revive a commented default;
        # otherwise append. Matching the commented form matters because the
        # shipped file documents every default as a comment.
        active = re.compile(rf"^(\s*){re.escape(key)}\s*=.*$", re.M)
        commented = re.compile(rf"^(\s*)#\s*{re.escape(key)}\s*=.*$", re.M)
        line = f"{key} = {value}"
        before = active.search(text) or commented.search(text)
        if active.search(text):
            text = active.sub(line, text, count=1)
        elif commented.search(text):
            text = commented.sub(line, text, count=1)
        else:
            text = text.rstrip("\n") + f"\n{line}\n"
        was = before.group(0).strip() if before else "<absent>"
        print(f"  {key:16s} {was}  ->  {line}")

    if text == original:
        print("no change; backup left at", backup)
        return 0
    open(PATH, "w").write(text)
    print(f"written. backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
