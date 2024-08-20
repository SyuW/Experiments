import numpy as np
import matplotlib.pyplot as plt


def logisticMap(x, mu):
    return 4 * mu * x * (1 - x)


def iterateMap(x, mu, num_iterations):
    out = x
    for i in range(num_iterations):
        out = logisticMap(out, mu)
    return out


def computeInvariantDensity(p, mu, num_iterations, bin_size):

    # equilibration first
    n_transient = 1000
    for i in range(n_transient):
        p = logisticMap(p, mu)
    
    # now collect a long trajectory and do binning
    trajectory = []
    bins = np.arange(0, 1, bin_size)
    for i in range(num_iterations):
        p = logisticMap(p, mu)
        trajectory.append(p)

    return trajectory


# plot the results of the iterated map
plot_iterates = False
if plot_iterates:
    mu = 1
    num_trials = 1000
    xrange = np.arange(0, 1, 0.001)
    out = iterateMap(xrange, mu, num_trials)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xrange, out, label="iterated map output")
    plt.plot(xrange, xrange, label="diagonal y=x")
    plt.title(f"Iterated logistic map for n={num_trials} iterations")
    plt.legend()
    plt.show()
    plt.clf()

# plot the invariant density of the attractor
plot_invariant_density = True
if plot_invariant_density:
    x_0 = 0.9
    bin_size = 0.001
    mu_val = 1.0
    data = computeInvariantDensity(x_0, mu=mu_val, num_iterations=10 ** 5, bin_size=bin_size)
    plt.title(rf"Computed invariant density with initial point $x_0$={x_0}, $\mu$={mu_val}")
    cusps = []
    val = 0.5
    for x in range(0, 10):
        val = logisticMap(val, mu_val)
        cusps.append(val)
    invariant_density_exact = lambda x: 1 / (np.pi * np.sqrt(x * (1 - x)))
    plt.plot(np.arange(0, 1, bin_size), invariant_density_exact(np.arange(0, 1, bin_size)), label=rf"Exact invariant density at $\mu$=1")
    plt.hist(data, bins=np.arange(0, 1, bin_size), density=True, label="Empirical density")

    plot_cusps = False
    if plot_cusps:
        for pos in cusps:
            plt.axvline(x=pos, color="red", label=rf"cusp at $x$ = {pos:.4f}", linestyle="--", alpha=0.6)

    plt.legend()
    plt.show()
    plt.clf()

# plot the bifurcation diagram of the attractor
plot_bifurcation = True
if plot_bifurcation:
    mu_min = 0.8
    mu_max = 1
    delta_mu = 0.00005
    bin_size = 0.0005
    n_transient = 100

    x_grid = np.arange(0, 1, bin_size)
    mu_grid = np.arange(mu_min, mu_max, delta_mu)

    diagram_grid = np.zeros((len(x_grid)-1, len(mu_grid)))

    for j, mu_val in enumerate(mu_grid):
        data = computeInvariantDensity(0.9, mu=mu_val, num_iterations=10**3, bin_size=bin_size)
        hist, _ = np.histogram(data, bins=x_grid, density=True)
        diagram_grid[:, j] = hist

plt.contourf(mu_grid, x_grid[1:], diagram_grid, norm="log")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$x$")
# plt.colorbar()

# plot the cusps as well
plot_cusps = False
if plot_cusps:
    for mu_val in mu_grid:
        x = 0.5
        cusps = []
        for i in range(8):
            x = logisticMap(x, mu=mu_val)
            cusps.append(x)
        plt.scatter(np.repeat(mu_val, 8), np.array(cusps), color="red")

plt.show()
plt.clf()

