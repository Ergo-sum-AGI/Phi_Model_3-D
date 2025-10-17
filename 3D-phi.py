import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For J5 quiver

PHI = (1 + np.sqrt(5)) / 2
ETA_TARGET = 0.809
LS_3D = [4, 8, 16]  # FSS sweep
BETA_C = np.log(1 + PHI) / 2
G_YUK = 1 / PHI
GAMMA_DEC = 1 / PHI**2
THETA_TWIST = np.pi / PHI

np.random.seed(42)

def phi_kernel_3d(L):
    x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0] = 1e-6
    kern = 1 / r**PHI * np.exp(-r / PHI)
    return kern / kern.sum()

def metropolis_step_3d(spins, beta, kernel, g_yuk, theta_twist):
    L = spins.shape[0]
    energy_old = - convolve(spins, kernel, mode='wrap') * spins
    spins_new = spins.copy()
    i, j, k = np.random.randint(0, L, 3)
    spins_new[i, j, k] *= -1
    energy_new = - convolve(spins_new, kernel, mode='wrap') * spins_new
    dE = energy_new[i, j, k] - energy_old[i, j, k] + g_yuk * np.random.randn()
    delta_sigma = spins_new[i, j, k] - spins[i, j, k]
    if i == 0 or i == L - 1:
        dE += theta_twist * np.sin(2 * np.pi * j / L) * delta_sigma * np.cos(2 * np.pi * k / L)
    accept = dE < 0 or np.random.rand() < np.exp(-beta * dE)
    spins[i, j, k] = spins_new[i, j, k] if accept else spins[i, j, k]
    return spins, accept

def wolff_cluster_3d(spins, beta):
    L = spins.shape[0]
    visited = np.zeros_like(spins, dtype=bool)
    flip = np.zeros_like(spins, dtype=bool)
    i, j, k = np.random.randint(0, L, 3)
    seed_spin = spins[i, j, k]
    stack = [(i, j, k)]
    visited[i, j, k] = True
    p_add = 1 - np.exp(-2 * beta)
    while stack:
        ci, cj, ck = stack.pop()
        flip[ci, cj, ck] = True
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni, nj, nk = (ci+di) % L, (cj+dj) % L, (ck+dk) % L
            if not visited[ni, nj, nk] and spins[ni, nj, nk] == seed_spin:
                if np.random.rand() < p_add:
                    visited[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
    spins[flip] *= -1
    return spins

def corr_3d(spins, r_max):
    L = spins.shape[0]
    center = L // 2
    corr = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    for dx in range(-r_max, r_max + 1):
        for dy in range(-r_max, r_max + 1):
            for dz in range(-r_max, r_max + 1):
                r = int(np.sqrt(dx**2 + dy**2 + dz**2))
                if 1 <= r <= r_max and all(0 <= center + d < L for d in (dx, dy, dz)):
                    corr[r] += spins[center, center, center] * spins[center + dx, center + dy, center + dz]
                    counts[r] += 1
    corr /= np.maximum(counts, 1)
    corr[1:] *= np.exp(-GAMMA_DEC * np.arange(1, r_max + 1))
    mask = counts[1:] > 0
    r_filtered = np.arange(1, r_max + 1)[mask]
    corr_filtered = np.abs(corr[1:][mask])
    return r_filtered, corr_filtered

def power_law_3d(r, A, eta_loc):
    return A / r**(1 + eta_loc)  # d=3: 1/r^{1+η}

eta_effs_3d = []
accepts_all_3d = []
for L_cur in LS_3D:
    spins = 2 * np.random.randint(0, 2, (L_cur, L_cur, L_cur)) - 1
    kernel_phi = phi_kernel_3d(L_cur)
    N_steps_L = max(500, L_cur**3 // 8)
    accepts_L = []
    for step in range(N_steps_L):
        spins, acc = metropolis_step_3d(spins, BETA_C, kernel_phi, G_YUK, THETA_TWIST)
        accepts_L.append(acc)
        if step % 10 == 0:
            spins = wolff_cluster_3d(spins, BETA_C)
    accepts_all_3d.append(np.mean(accepts_L))
    r, G = corr_3d(spins, r_max=L_cur//4)
    if len(r) > 10:
        try:
            popt, _ = curve_fit(power_law_3d, r[::2], G[::2], p0=[1, 0.25], maxfev=5000)
            eta_effs_3d.append(popt[1])
        except:
            eta_effs_3d.append(np.nan)
    else:
        eta_effs_3d.append(np.nan)

eta_effs_3d = np.array(eta_effs_3d)
x_3d = 1 / np.array(LS_3D)
mask_3d = ~np.isnan(eta_effs_3d)
if np.sum(mask_3d) > 2:
    popt_fss_3d, pcov_fss_3d = curve_fit(fss_omega, x_3d[mask_3d], eta_effs_3d[mask_3d], p0=[0.8, 0.5])
    eta_extrap_3d = popt_fss_3d[0]
    eta_err_3d = np.sqrt(pcov_fss_3d[0,0])
else:
    eta_extrap_3d = np.nan
    eta_err_3d = np.nan

print(f"3D Extrapolated η: {eta_extrap_3d:.3f} ± {eta_err_3d:.3f} (target {ETA_TARGET:.3f})")

# Plot (2x3: spins slice, G(r), FSS, β (reuse 2D), kernel slice, J5 quiver)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(spins[:, :, L_cur//2], cmap='RdBu')
axs[0, 0].set_title(f'3D Spins Slice (L={L_cur})')
r_plot, G_plot = corr_3d(spins, r_max=L_cur//4)
axs[0, 1].semilogy(r_plot, G_plot, 'o-', label=f'η_eff={eta_effs_3d[-1]:.3f}')
axs[0, 1].set_title('3D G(r)'); axs[0, 1].legend()
axs[0, 2].semilogy(LS_3D, eta_effs_3d, 'o-', label='η_eff(L)')
axs[0, 2].plot(1/x_3d, fss_omega(x_3d, *popt_fss_3d), '--', label=f'η_∞={eta_extrap_3d:.3f}')
axs[0, 2].set_title('3D FSS'); axs[0, 2].legend()
# β reuse from 2D or skip
axs[1, 1].imshow(kernel_phi[:, :, L_cur//2], cmap='hot')
axs[1, 1].set_title('3D Kernel Slice')
# J5
from scipy.ndimage import gaussian_filter
J5 = gaussian_filter(np.gradient(spins, axis=0) - np.gradient(spins, axis=1) + np.gradient(spins, axis=2), sigma=1)
axs[1, 2].quiver(J5[::4, ::4, L_cur//2])  # Subsample
axs[1, 2].set_title('J5 Chiral Currents')
plt.tight_layout()
plt.savefig('3d_phi_proto.png', dpi=300)
plt.show()

print('*** 3D Proto Run Complete! Plot: 3d_phi_proto.png')