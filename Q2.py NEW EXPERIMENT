import numpy as np, matplotlib.pyplot as plt

N = np.array([3, 4, 7, 9, 12])
gamma, total = 4, 336
Q = np.sqrt(3 * N)                              # co-channel reuse ratio
ch = total / N                                  # channels per cell
si_db = 10 * np.log10(Q**gamma / 6)             # first-tier S/I in dB

print("N    Q      ch/cell   S/I(dB)")
for n, q, c, s in zip(N, Q, ch, si_db):
    print(f"{n:<4d} {q:5.2f} {c:7.1f}           {s:7.2f}")

fig, ax1 = plt.subplots()
ax1.plot(N, si_db, 'o-b'); ax1.set_xlabel('Cluster size N')
ax1.set_ylabel('S/I (dB)', color='b')
ax2 = ax1.twinx(); ax2.plot(N, ch, 's--r')
ax2.set_ylabel('Channels per cell', color='r')
plt.title('Capacity vs interference trade-off'); plt.savefig('reuse.png', dpi=150)
