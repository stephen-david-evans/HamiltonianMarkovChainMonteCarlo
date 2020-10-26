"""Hamiltonian Markov Chain Monte Carlo
"""
import tqdm
import numpy as np
import scipy.stats as st
from scipy.misc import derivative


def leapfrog(position0, momentum0, fprime, path_len, step_size):
    """leapfrog integrator"""
    position, momentum = position0.copy(), momentum0.copy()

    for _ in range(int(path_len / step_size)):
        momentum -= 0.5 * step_size * fprime(position)
        position += step_size * momentum
        momentum -= 0.5 * step_size * fprime(position)

    return position, -momentum


def hmc(
    func,
    x0,
    n=1000,
    burn_fraction=0.1,
    seed=0,
    derivative_step_size=1.0,
    derivative_order=3,
    integrator_step_size=0.1,
    integrator_path_len=1.0,
    progress_bar=False,
):
    """Hamiltonian Markov Chain Monte Carlo sampling with numerical derivative"""

    n = int(n)

    def f(x):
        """return negative log probability of input function"""
        return -np.log(func(x))

    def fprime(x):
        """return numerical derivative of negative log prob function"""
        return derivative(f, x, dx=derivative_step_size, order=derivative_order)

    # fix a seed object - to allow repeatable uses
    seed = np.random.RandomState(seed)

    # pre-calculate all momentum samples
    momentum = st.norm(0, 1)
    momentum_samples = momentum.rvs(size=(n, x0.shape[0]), random_state=seed)

    # pre-calculate all acceptance ratios
    log_ratios = np.log(seed.uniform(size=n))

    # storage for samples - will be returned as np.array
    samples = [x0]

    if progress_bar:
        iterable = tqdm.tqdm(enumerate(momentum_samples), total=n)
    else:
        iterable = enumerate(momentum_samples)

    for i, momentum0 in iterable:
        position1, momentum1 = leapfrog(
            samples[-1], momentum0, fprime, integrator_path_len, integrator_step_size
        )

        e0 = momentum.logpdf(momentum0).sum() - f(samples[-1])
        e1 = momentum.logpdf(momentum1).sum() - f(position1)

        new_sample = position1 if log_ratios[i] < e1 - e0 else samples[-1]
        samples.append(new_sample)

    return np.array(samples[int(burn_fraction * n):])
