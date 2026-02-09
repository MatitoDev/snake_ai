import struct

path = "qtable.bin"
with open(path, "rb") as f:
    states, actions = struct.unpack(">ii", f.read(8))
    alpha, gamma, eps, eps_min, eps_decay = struct.unpack(">ddddd", f.read(8*5))
    print(states, actions, alpha, gamma, eps, eps_min, eps_decay)

    # erste 10 states anzeigen
    for s in range(min(10, states)):
        row = [struct.unpack(">d", f.read(8))[0] for _ in range(actions)]
        print(s, row)

