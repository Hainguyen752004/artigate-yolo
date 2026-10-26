"""Download MALUNet's model file at the pinned commit (the repository has no licence, so it is not committed).

    python scripts/fetch_malunet.py
writes third_party/malunet/malunet.py and checks its sha256; see third_party/malunet/README.md.
"""
import argparse
import hashlib
from pathlib import Path
import ssl
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e184b47de99b3fda6b7880fcf3ff99b806308d2b"
URL = f"https://raw.githubusercontent.com/JCruan519/MALUNet/{COMMIT}/models/malunet.py"
SHA256 = "c7c1bacd4f7b434670c97416e490316edeac9b1ab19ec626a070ac6d73308dc0"
TARGET = ROOT / "third_party" / "malunet" / "malunet.py"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    if args.out.is_file() and hashlib.sha256(args.out.read_bytes()).hexdigest() == SHA256:
        print(f"{args.out} already present (sha256 ok)")
        return args.out
    try:  # certifi's bundle avoids broken entries in the Windows certificate store
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    with urllib.request.urlopen(URL, timeout=60, context=context) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch for {URL}: {digest}, expected {SHA256}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"wrote {args.out} (MALUNet {COMMIT[:7]}, sha256 ok)")
    return args.out


if __name__ == "__main__":
    main()
