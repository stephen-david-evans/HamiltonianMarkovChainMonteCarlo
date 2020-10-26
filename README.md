# HamiltonianMarkovChainMonteCarlo
Basic implementation of Hamiltonian Markov Chain Monte Carlo with numerical derivatives

## Example

    import matplotlib.pyplot as plt

    mu = np.sqrt(2.2)
    sigma = np.sqrt(3.5)
    x0 = np.array([0.5])

    def test(x):
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2.0)

    samples = hmc(test, x0, n=50000, progress_bar=True)

    plt.figure(tight_layout=True)
    ax = plt.subplot()

    res = st.probplot(samples[:, 0], plot=ax, dist=st.norm, sparams=(mu, sigma))
    print(res[1])

    plt.show()
