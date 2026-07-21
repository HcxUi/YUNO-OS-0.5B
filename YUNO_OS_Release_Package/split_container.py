import os
import glob
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNO_FILE = os.path.join(RELEASE_DIR, "yuno-os-v0.5.0.yuno")
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB per part file

def split_yuno_container():
    if not os.path.exists(YUNO_FILE):
        print(f"[-] Error: {YUNO_FILE} does not exist.")
        return

    print(f"[*] Splitting '{YUNO_FILE}' into {CHUNK_SIZE // (1024*1024)}MB GitHub-compatible chunk parts...")

    # Remove any existing old part files first
    old_parts = glob.glob(os.path.join(RELEASE_DIR, "yuno-os-v0.5.0.yuno.part*"))
    for p in old_parts:
        try:
            os.remove(p)
            print(f"  - Cleaned up old part: {os.path.basename(p)}")
        except Exception as e:
            print(f"  - Failed to remove {p}: {e}")

    part_idx = 0
    with open(YUNO_FILE, "rb") as f_in:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            part_filename = f"yuno-os-v0.5.0.yuno.part{part_idx:02d}"
            part_path = os.path.join(RELEASE_DIR, part_filename)
            with open(part_path, "wb") as f_out:
                f_out.write(chunk)
            print(f"  ✓ Created chunk {part_filename} ({len(chunk):,} bytes)")
            part_idx += 1

    print(f"[+] Container successfully split into {part_idx} chunks!")

if __name__ == "__main__":
    split_yuno_container()
