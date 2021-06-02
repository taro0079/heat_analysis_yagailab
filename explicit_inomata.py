# 陽解法ver6

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation


# 定数
z = 8960.0  # [kg/m^3]
C = 385.0  # [J/kg*K]
k = 398.0  # [W/m*K]
T0_r = 300.0  # [度]
T0_l = 20.0  # [度]
L = 0.65  # [m]
yousosu_x = 66
te = 10000  # [s]
dt = 0.1
dx = 0.01

R = k / (C * z)
yousosu_t = int(te / dt)
S = R * (dt / (dx ** 2))

# 行列の用意、Ty=A*Tx
a = []
b = []
for j in range(66):
    a.append(1 - 2 * S)

for j in range(65):
    b.append(S)

x1 = np.diag(a, k=0)
x2 = np.diag(b, k=1)
x3 = np.diag(b, k=-1)

A = x1 + x2 + x3
Tx = np.full((yousosu_x, 1), T0_l)

Ty = np.zeros_like(Tx)

# 境界条件
Tx[0, 0] = T0_r


# グラフ作成の用意
ims = []
fig = plt.figure()
x = np.linspace(0, 66, 66)

B = np.zeros_like(Tx)
B[0] = S * T0_r
B[-1] = S * T0_l

# 繰り返し
for n in range(1, yousosu_t + 1):
    Tx = Ty.copy()
    # for i in range(1, yousosu_x - 1):
    #    Ty[i, 0] = (
    #        Tx[i - 1, 0] * A[i, i - 1] + Tx[i, 0] * A[i, i] + Tx[i + 1, 0] * A[i, i + 1]
    #    )

    # for i in range(1, yousosu_x - 1):
    #    Tx[i, 0] = Ty[i, 0]

    # Tx[0, 0] = T0_r
    Ty = np.dot(A, Tx) + B
print(Ty)
# if n % 100 == 0:
#     im = plt.plot(x, Tx, "r")
#     ims.append(im)
#     print(Tx[i, 0], n)

# ani = animation.ArtistAnimation(fig, ims, interval=100)
# plt.show()
