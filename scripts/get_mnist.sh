set -eu

DATA_DIR="${DATA_DIR:-data/mnist}"
MIRRORS="
https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data
https://ossci-datasets.s3.amazonaws.com/mnist
"

# file name, exact size in bytes after decompression
FILES="
train-images-idx3-ubyte 47040016
train-labels-idx1-ubyte 60008
t10k-images-idx3-ubyte 7840016
t10k-labels-idx1-ubyte 10008
"

mkdir -p "$DATA_DIR"

echo "$FILES" | while read -r name size; do
    [ -z "$name" ] && continue

    if [ -f "$DATA_DIR/$name" ] && [ "$(wc -c < "$DATA_DIR/$name")" -eq "$size" ]; then
        echo "ok       $name (already here)"
        continue
    fi

    got=0
    for mirror in $MIRRORS; do
        if curl -fsSL --max-time 120 -o "$DATA_DIR/$name.gz" "$mirror/$name.gz" 2>/dev/null; then
            gunzip -f "$DATA_DIR/$name.gz" 2>/dev/null || { rm -f "$DATA_DIR/$name.gz" "$DATA_DIR/$name"; continue; }
            if [ "$(wc -c < "$DATA_DIR/$name")" -eq "$size" ]; then
                echo "ok       $name ($size bytes, from $mirror)"
                got=1
                break
            fi
            echo "wrong size from $mirror, trying the next mirror" >&2
            rm -f "$DATA_DIR/$name"
        fi
    done

    if [ "$got" -eq 0 ]; then
        echo "FAILED   $name: no mirror gave a file of exactly $size bytes" >&2
        exit 1
    fi
done

echo "done: $DATA_DIR"