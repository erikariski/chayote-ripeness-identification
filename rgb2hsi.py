from math import acos, pi, sqrt

def RGB2HSI(colour):
    (R, G, B) = colour

    r = R / ((R + 0.000001) + (G + 0.000001) + (B + 0.000001))
    g = G / ((R + 0.000001) + (G + 0.000001) + (B + 0.000001))
    b = B / ((R + 0.000001) + (G + 0.000001) + (B + 0.000001))

    num = 0.5 * ((r - g) + (r - b))
    den = sqrt((r - g) ** 2 + (r - b) * (g - b))
    h = acos(num / (den + 0.0000001))

    if b <= g:
        h = h
    else:
        h = 2 * pi - h

    H = float(h * 180 / pi)

    S = float(1 - 3 * min(r, g, b))

    I = float(R + G + B) / (3 * 255)

    return (H, S, I)
