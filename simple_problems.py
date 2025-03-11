import numpy as np

def quadratic(seed):
    """Simple quadratic function: y = 2*x^2 + 3*x + 1"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(50, 1))
    x = X[:, 0]
    y = 2 * x**2 + 3 * x + 1
    return X, y

def cubic(seed):
    """Simple cubic function: y = x^3 - 2*x^2 + x"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(50, 1))
    x = X[:, 0]
    y = x**3 - 2 * x**2 + x
    return X, y

def simple_rational(seed):
    """Simple rational function: y = 5 / (1 + x^2)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-4, 4, size=(50, 1))
    x = X[:, 0]
    y = 5 / (1 + x**2)
    return X, y

def simple_physics(seed):
    """Simple physics equation: kinetic energy E = 0.5 * m * v^2
    where m=mass, v=velocity"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.1, 10, size=(50, 2))  # [mass, velocity]
    m = X[:, 0]
    v = X[:, 1]
    y = 0.5 * m * v**2
    return X, y

def simple_trig(seed):
    """Simple trigonometric function: y = 2*sin(x) + cos(2*x)"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-np.pi, np.pi, size=(50, 1))
    x = X[:, 0]
    y = 2 * np.sin(x) + np.cos(2 * x)
    return X, y


# List of simple problems
SIMPLE_PROBLEMS = [
    quadratic,
    cubic,
    simple_rational,
    simple_physics,
    simple_trig
]