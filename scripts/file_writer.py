import sys, os, base64

def write_file(target, b64_content):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'wb') as f:
        f.write(base64.b64decode(b64_content))
    print(f"Wrote {os.path.getsize(target)} bytes to {target}")

if __name__ == '__main__':
    write_file(sys.argv[1], sys.argv[2])
