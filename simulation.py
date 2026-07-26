import math

FARADAY = 96485.33212  # C/mol
GAS_CONSTANT = 8.314  # J/mol/K
TEMPERATURE = 298.15  # K


def simulate_cv(E_start=-0.5, E_reverse=0.5, E0=0.0, scan_rate=0.1,
                area=1e-4, diffusion=1e-5, concentration=1e-3,
                n_electron=1, steps=200):
    dt = abs(E_reverse - E_start) / scan_rate / (steps / 2)
    dx = math.sqrt(2.0 * diffusion * dt * 1.1)
    n_grid = 50
    c_o = [concentration for _ in range(n_grid)]
    c_r = [0.0 for _ in range(n_grid)]
    potentials = []
    currents = []
    seq = []
    half = steps // 2
    for i in range(half):
        frac = i / float(half)
        seq.append(E_start + (E_reverse - E_start) * frac)
    for i in range(half, steps):
        frac = (i - half) / float(half)
        seq.append(E_reverse - (E_reverse - E_start) * frac)
    for E in seq:
        ratio = math.exp(n_electron * FARADAY * (E - E0) / (GAS_CONSTANT * TEMPERATURE))
        c_o[0] = concentration * ratio / (1 + ratio)
        c_r[0] = concentration - c_o[0]
        new_o = c_o[:]
        new_r = c_r[:]
        factor = diffusion * dt / (dx * dx)
        for j in range(1, n_grid - 1):
            new_o[j] = c_o[j] + factor * (c_o[j + 1] - 2 * c_o[j] + c_o[j - 1])
            new_r[j] = c_r[j] + factor * (c_r[j + 1] - 2 * c_r[j] + c_r[j - 1])
        new_o[-1] = concentration
        new_r[-1] = 0.0
        c_o, c_r = new_o, new_r
        flux = diffusion * (c_o[1] - c_o[0]) / dx
        current = n_electron * FARADAY * area * flux
        potentials.append(E)
        currents.append(current)
    return potentials, currents
