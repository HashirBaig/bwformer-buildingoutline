import sys
import numpy as np

def main(path):
    obj = np.load(path, allow_pickle=True)
    try:
        data = obj.item()
    except Exception:
        data = obj
    print(f"Loaded: {path}")
    print(f"Top-level type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f" - {k}: ndarray shape={v.shape} dtype={v.dtype}")
            else:
                print(f" - {k}: type={type(v).__name__}")
    elif isinstance(data, np.ndarray):
        print(f"Array shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == object:
            flat = list(data.flat)
            types = {type(x).__name__ for x in flat[: min(10, len(flat))]}
            print(f"Sample element types: {types}")
    else:
        print(repr(data))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_npy.py <file.npy>")
        sys.exit(1)
    main(sys.argv[1])

