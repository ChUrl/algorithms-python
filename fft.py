#!/usr/bin/env python3

import numpy as np

from rich.traceback import install
install()

# We want to solve the FFT and inverse FFT in an easily accessible way, not necessarily efficient.
#
# The coefficient representation:
# - Every polynomial of degree d is of the form P_d(x) = c_0 * x**0 + c_1 * x**1 + ... + c_d * x**d
# - There exists a bijective mapping between the set of (d + 1)-tuples of coefficients and the set of polynomials
# - If we store a list of coefficients where the index corresponds to the degree we have a very simple representation of a polynomial
# - Example: P_2(x) = x**2 + 5 has the coefficient representation [5, 0, 1]
#
# The value representation:
# - There exists a surjective mapping between the set of (d + 1)-tuples of points from the polynomial's image to the set of polynomials
# - Example: P_2(x) = x**2 could have the value representation [(-1, 1), (1, 1)] or [(-2, 4), (2, 4)]
#
# Roots of unity:
# - The n-th roots of unity are all complex numbers satisfying the equation z**n = 1.
# - They can be visualized as n equally spaced points on the complex unit circle
#   - In the real domain the unit circle is defined by the constraint x**2 + y**2 = 1, meaning every point on the real unit circle has the distance 1 from the origin
#     Different formulation: We have a point (x, y) on the real unit circle with the angle p: x = cos(p), y = sin(p) and we know that cos(p)**2 + sin(p)**2 = 1, which is the distance to the origin (pythagorean theorem)
#   - In the complex domain the pythagorean theorem works the same: We have a complex number c = (c_real, c_img) on the complex unit circle with the angle p: c_real = cos(p), c_img = sin(p)
#     If we take Eulers formula: e**(i * p) = cos(p) + i * sin(p), we can easily define the r-th root of the n-th roots of unity:
#     - The angle between the n-th roots is (2pi)/n, because all n roots are equally spaced
#     - The angle of the r-th root can be written as (2pi * r)/n
#     - The r-th root can thus be written as cos((2pi * r)/n) + i * sin((2pi * r)/n), which is equal to e**((i * 2pi * r)/n)
#     - The "distance from the origin = 1" constraint holds (obviously): e**((i * 2pi * r)/n) = (e**((i * pi * r)/n))**2 = cos((2pi * r)/n)**2 + i * sin((2pi * r)/n)**2 = 1
# - With this visualization it is also easy to define them in a closed form term: w**r = e**((i * 2pi * r)/n) (w**r is the r-th of the n-th roots of unity)
# - It is also clear that the roots always come in +/- pairs, because if they are equally spaced on the complex unit circle every root is connected to another root by a line through the origin
#
# Calculating the value representation (FFT):
# - The naive approach is to evaluate polynomial of degree d in (d + 1) distinct points,
#   but this would have a runtime of O(d**2), as every evaluation has a runtime of O(d)
# - To lower the runtime we can make use of a polynomial's symmetry: Even polynomials are symmetric regarding the imaginary axis, odd polynomials are symmetric regarding the origin
# - Every polynomial can be split into a sum of an even and an odd polynomial, so if we do this a single time, we have a runtime of O((d/2) * d), which is still quadratic
#   - We split e.g. P_3(x) = 3 * x**3 + x**2 + 2 * x + 5 into P_3_even(x) = x + 5 and P_3_odd(x) = 3 * x + 2,
#     so we get the alternative representation P_3(x) = P_3_even(x**2) + x * P_3_odd(x**2) = x**2 + 5 + x * (3 * x**2 + 2) = 3 * x**3 + x**2 + 2 * x + 5.
#     This way we can always make use of the symmetry of even polynomials
#   - We only need to evaluate the polynomial in d/2 points, because by using the symmetry every point can be transformed into its counterpart "on the other side":
#     If we take the polynomial P_2(x) = x**2, the value representation could be [(-1, 1), (1, 1)], where (-1, 1) was inferred from one evaluation at x = 1
# - In general, we can write the value representation for a polynomial of 2nd degree as [(-x, P_2(x)), (x, P_2(x))]
# - To lower the runtime further one idea is obvious: Recursively split the polynomials to reach a runtime of O(d * log(d))
#   - This raises a problem though: To infer all needed points from the polynomials image we always have to evaluate the polynomial in +/- pairs, e.g. x = 1 and x = -1.
#     If we try to evaluate the split polynomial (P_3_even(x**2) and P_3_odd(x**2)) we can no longer choose arbitrary values, as x**2 is always positive in the real domain
#   - The solution is to use the roots of unity as evaluation points :
#     - If we want to calculate the FFT of P_2(x) = x**2 with P_2_even(x**2) = x and P_2_odd(x**2) = 0 we choose the 4th roots of unity as evaluation points for P_2(x)
#       (4th because we need at least 3 points for the value representation of a degree 2 polynomial), and the 2nd roots of unity as evaluation points for P_2_even(x**2) and P_2_odd(x**2)
# - The last step is to combine the split value representations back into one:
#
# Calculating the coefficient representation (IFFT):


# Main Algorithms


def FFT_recursive(n: int, coefficient_repr: list):
    if n == 1:
        return coefficient_repr # In this case we already have the value representation as the function is constant

    # Split the function in even and odd parts of degree n/2
    even_coefficients = coefficient_repr[0::2]
    odd_coefficients = coefficient_repr[1::2]

    # This step is the important part: We recursively apply FFT to reach the runtime of O(n log n). The result are two value representations evaluated in the (n/2)ths roots of unity.
    even_values = FFT_recursive(int(n/2), even_coefficients)
    odd_values = FFT_recursive(int(n/2), odd_coefficients)

    w = np.exp(1j * 2 * np.pi / n) # w**r is our r-th root of the n-th roots of unity

    value_repr = [0] * n
    for i in range(int(n/2)):
        # Note that the odd values have to be multiplied by the evaluation point because we extracted one "x" from the term to get an even polynomial again
        value_repr[i] = even_values[i] + w**i * odd_values[i] # Evaluate the even part of our polynomial (symmetry regarding x-axis => +, only the evaluation point is negative)
        value_repr[i + int(n/2)] = even_values[i] - w**i * odd_values[i] # Evaluate the odd part of our polynomial (symmetry regarding origin => -, the evaluation point and evaluation value are negative)

    return value_repr

def IFFT_recursive(n: int, value_repr: list):
    if n == 1:
        return value_repr

    even_values = value_repr[0::2]
    odd_values = value_repr[1::2]

    even_coefficients = IFFT_recursive(int(n/2), even_values)
    odd_coefficients = IFFT_recursive(int(n/2), odd_values)

    w = np.exp(-1j * 2 * np.pi / n) # The DFT can be inverted by just conjugating the exponants and normalizing everything later by 1/n

    coefficient_repr = [0] * n
    for i in range(int(n/2)):
        coefficient_repr[i] = even_coefficients[i] + w**i * odd_coefficients[i]
        coefficient_repr[i + int(n/2)] = even_coefficients[i] - w**i * odd_coefficients[i]

    return coefficient_repr


# Helper functions


def power_of_two(n):
    "Find the smallest power of two that is larger than n."
    power = 0
    while 2**power < n:
        power += 1
    return power

def pad_coefficients(coefficient_repr):
    "Transform a coefficient representation to a representation of higher degree (FFT handles polynomials with degrees of powers of two)."
    power = power_of_two(len(coefficient_repr))
    for _ in range(2**power - len(coefficient_repr)):
        coefficient_repr += [(0+0j)]
    return coefficient_repr

# FFT basically converts from the frequency domain to the time/signal domain
# The coefficient representation
def FFT(coefficient_repr):
    "Convert a polynomial from its coefficient representation to its value representation."
    n = len(coefficient_repr)
    return FFT_recursive(n, coefficient_repr)

# FFT basically converts from the time/signal domain to the frequency domain
# The value representation is the signal: A lot of values at different points in time
def IFFT(value_repr):
    "Convert a polynomial from its coefficient representation to its value representation."
    n = len(value_repr)
    unnormalized = IFFT_recursive(n, value_repr)

    # The second part of the DFT matrix inversion: Normalize by 1/n
    normalized = [(0+0j)] * n
    for i in range(n):
        normalized[i] = unnormalized[i] / n

    return normalized


# FFT:
print("FFT:")
coefficient_repr = pad_coefficients([(0+0j), (0+0j), (1+0j), (0+0j)]) # P_2(x) = x**2
value_repr = FFT(coefficient_repr)
print("Coefficient Representation:", coefficient_repr)
print("Value Representation:", value_repr)

# IFFT:
print("IFFT:")
coefficient_repr = IFFT(value_repr)
print("Value Representation:", value_repr)
print("Coefficient Representation:", coefficient_repr)
