import struct
import sys


def read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Unexpected EOF: wanted {n} bytes, got {len(b)}")
    return b


def read_i32(f) -> int:
    return struct.unpack(">i", read_exact(f, 4))[0]


def read_f64(f) -> float:
    return struct.unpack(">d", read_exact(f, 8))[0]


def read_f64_array(f, n: int):
    if n < 0:
        raise ValueError("Negative length")
    if n == 0:
        return []
    return list(struct.unpack(f">{n}d", read_exact(f, 8 * n)))


def read_matrix(f, rows: int, cols: int):
    flat = read_f64_array(f, rows * cols)
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def read_network(f, label: str):
    input_size = read_i32(f)
    h1 = read_i32(f)
    h2 = read_i32(f)
    out = read_i32(f)

    w1 = read_matrix(f, h1, input_size)
    b1 = read_f64_array(f, h1)

    w2 = read_matrix(f, h2, h1)
    b2 = read_f64_array(f, h2)

    w3 = read_matrix(f, out, h2)
    b3 = read_f64_array(f, out)

    print(f"\n{label}")
    print(f"  dims: input={input_size}, h1={h1}, h2={h2}, out={out}")
    print(f"  w1[0][0:5] = {w1[0][0:5] if h1 and input_size else []}")
    print(f"  b1[0:5]    = {b1[0:5]}")
    print(f"  w2[0][0:5] = {w2[0][0:5] if h2 and h1 else []}")
    print(f"  b2[0:5]    = {b2[0:5]}")
    print(f"  w3[0][0:5] = {w3[0][0:5] if out and h2 else []}")
    print(f"  b3         = {b3}")

    return {
        "input": input_size,
        "h1": h1,
        "h2": h2,
        "out": out,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dqn_weights.bin"

    with open(path, "rb") as f:
        alpha = read_f64(f)
        gamma = read_f64(f)
        eps = read_f64(f)
        eps_min = read_f64(f)
        eps_decay = read_f64(f)
        target_update_freq = read_i32(f)
        update_counter = read_i32(f)

        print("Header")
        print(f"  alpha={alpha}")
        print(f"  gamma={gamma}")
        print(f"  epsilon={eps}")
        print(f"  epsilonMin={eps_min}")
        print(f"  epsilonDecay={eps_decay}")
        print(f"  targetUpdateFrequency={target_update_freq}")
        print(f"  updateCounter={update_counter}")

        read_network(f, "Online network")
        read_network(f, "Target network")

        leftover = f.read()
        if leftover:
            print(f"\nNote: {len(leftover)} trailing bytes remain (file format mismatch?)")


if __name__ == "__main__":
    main()

