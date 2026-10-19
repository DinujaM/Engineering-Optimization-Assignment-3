import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11, 'axes.grid': True})

# --- Figure 1: Q1(a) Lagrangian curves ---
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.linspace(-1, 3, 400)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for mu, col in zip([0, 1, 2, 3], colors):
    L = x**2 + mu * (1 - x)
    x_min, L_min = mu / 2.0, mu - (mu**2) / 4.0
    ax.plot(x, L, label=f'$\\mu = {mu}$', color=col, linewidth=2)
    ax.plot(x_min, L_min, 'o', color=col, markersize=7)
    ax.annotate(f'$x({mu})={x_min}$', (x_min, L_min), textcoords="offset points", 
                xytext=(0, -18), ha='center', fontsize=9, fontweight='bold', color=col)

ax.axvline(1.0, color='gray', linestyle='--', label='Boundary $x=1$')
ax.set_title('Q1(a): Lagrangian $L(\\mu, x) = x^2 + \\mu(1-x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$L(\\mu, x)$')
ax.legend()
plt.tight_layout()
plt.savefig('fig_q1a.png', dpi=300)
plt.show()

# --- Figure 2: Q1(b) Dual function ---
fig, ax = plt.subplots(figsize=(6.5, 4))
mu = np.linspace(0, 4, 400)
ax.plot(mu, mu - (mu**2)/4.0, 'b-', linewidth=2, label='$d(\\mu) = \\mu - \\mu^2/4$')
ax.plot(2, 1, 'ro', markersize=8, label='Optimal $(\\mu^*=2, d^*=1)$')
ax.annotate('$(\\mu^*=2, d^*=1)$', xy=(2, 1), xytext=(2.2, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
ax.set_title('Q1(b): Dual Function $d(\\mu)$ vs $\\mu$')
ax.set_xlabel('$\\mu$')
ax.set_ylabel('$d(\\mu)$')
ax.legend()
plt.tight_layout()
plt.savefig('fig_q1b.png', dpi=300)
plt.show()

# --- Figure 3: Q1(c) Primal function ---
fig, ax = plt.subplots(figsize=(6.5, 4))
x_feas = np.linspace(1, 3, 300)
ax.plot(x_feas, x_feas**2, 'b-', linewidth=2.5, label='$g(x) = x^2$ ($x \\geq 1$)')
ax.axvspan(-1, 1, color='red', alpha=0.15, label='Infeasible region ($g(x)=+\\infty$)')
ax.axvline(1, color='red', linestyle='--', linewidth=2)
ax.plot(1, 1, 'go', markersize=8, label='Primal minimum $(x^*=1, p^*=1)$')
ax.set_xlim([-1, 3])
ax.set_ylim([0, 10])
ax.set_title('Q1(c): Primal Objective Function $g(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$g(x)$')
ax.legend()
plt.tight_layout()
plt.savefig('fig_q1c.png', dpi=300)
plt.show()

# --- Figure 4: Q2(a) Saddle point cross-sections ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
mu_vals = np.linspace(0, 4, 200)
ax1.plot(mu_vals, np.ones_like(mu_vals), 'm-', linewidth=2, label='$L(\\mu, x^*) = 1$')
ax1.plot(2, 1, 'ro', markersize=8, label='$L(\\mu^*, x^*) = 1$')
ax1.set_title('$L(\\mu, x^*)$ vs $\\mu$')
ax1.set_xlabel('$\\mu$')
ax1.set_ylabel('$L(\\mu, x^*)$')
ax1.set_ylim([0, 2.5])
ax1.legend()

x_vals = np.linspace(-1, 3, 200)
ax2.plot(x_vals, (x_vals - 1)**2 + 1, 'g-', linewidth=2, label='$L(\\mu^*, x) = (x-1)^2 + 1$')
ax2.plot(1, 1, 'ro', markersize=8, label='$L(\\mu^*, x^*) = 1$')
ax2.set_title('$L(\\mu^*, x)$ vs $x$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$L(\\mu^*, x)$')
ax2.legend()
plt.tight_layout()
plt.savefig('fig_q2a.png', dpi=300)
plt.show()

# --- Figure 5: Q3(b) Lower envelope ---
fig, ax = plt.subplots(figsize=(7, 4.5))
mu_q3 = np.linspace(0, 4, 300)
L_x0 = mu_q3
L_x1 = np.ones_like(mu_q3)
L_x2 = 4 - mu_q3

ax.plot(mu_q3, L_x0, ':', label='$x = 0: L(\\mu, 0) = \\mu$')
ax.plot(mu_q3, L_x1, '--', label='$x = 1: L(\\mu, 1) = 1$')
ax.plot(mu_q3, L_x2, '-.', label='$x = 2: L(\\mu, 2) = 4 - \\mu$')
lower_env = np.minimum(np.minimum(L_x0, L_x1), L_x2)
ax.plot(mu_q3, lower_env, 'k-', linewidth=3, label='Envelope $\\min_{x \\in \\{0,1,2\\}} L(\\mu, x)$')
ax.plot(mu_q3, mu_q3 - (mu_q3**2)/4.0, 'r--', alpha=0.6, label='True $d(\\mu) = \\mu - \\mu^2/4$')
ax.set_title('Q3(b): Affine Functions & Lower Envelope Approximation')
ax.set_xlabel('$\\mu$')
ax.set_ylabel('$L(\\mu, x)$')
ax.legend()
plt.tight_layout()
plt.savefig('fig_q3b.png', dpi=300)
plt.show()