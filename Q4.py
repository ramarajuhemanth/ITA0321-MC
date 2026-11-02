import numpy as np
from scipy.linalg import hadamard

W = hadamard(4)                        # 4 orthogonal Walsh codes (rows), +/-1
bits1 = np.array([1, -1, 1, 1])        # user 1 data (+/-1)
bits2 = np.array([-1, -1, 1, -1])      # user 2 data

def spread(bits, code):                # each bit multiplied by the whole code
    return np.concatenate([b * code for b in bits])

tx = spread(bits1, W[1]) + spread(bits2, W[2])            # channel adds signals

def despread(rx, code):
    chips = rx.reshape(-1, len(code))
    return np.sign(chips @ code / len(code))

print("User1 recovered:", despread(tx, W[1]), "sent:", bits1)
print("User2 recovered:", despread(tx, W[2]), "sent:", bits2)
print("Wrong code W3 correlation:", (tx.reshape(-1, 4) @ W[3]) / 4)   # all zeros
