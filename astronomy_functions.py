import numpy as np


def apply_extinction_correction(wavelengths, flux, R_v, A_v = 1):
    """
    Apply extinction correction to a spectrum based on the Cardelli, Clayton, & Mathis (CCM) extinction law.

    The CCM extinction law describes how the interstellar extinction affects the spectrum of light passing through
    a dust-filled medium. It provides a relationship between the wavelength-dependent extinction, represented by A_lambda,
    and the visual extinction magnitude, A_v, and the total-to-selective extinction ratio, R_v.

    Args:
        wavelengths (array-like): Array of wavelengths in Angstrom.
        R_v (float): Total-to-selective extinction ratio.
        A_v (float, optional): Visual extinction magnitude. Default value is 1.

    Returns:
        corrected_flux (array-like): Array of corrected flux values corresponding to the input wavelengths, after the extinction is calculated and applied

    Notes:
        The CCM extinction law is divided into several wavelength regions with different functional forms.
        The coefficients for these functional forms have been pre-determined and are used to calculate the extinction values.

        The input wavelengths should be between 1000 and 10000 Angstrom

    References:
        Cardelli, J. A., Clayton, G. C., & Mathis, J. S. (1989). The relationship between infrared,
        optical, and ultraviolet extinction. The Astrophysical Journal, 345, 245-256.
        https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract

    """

    # CCM coefficients
    a = np.zeros_like(wavelengths)
    b = np.zeros_like(wavelengths)
    wavelengths = wavelengths * (10**-4)  # Convert wavelengths to microns

    # Define wavelength regions
    x = (1.0 / wavelengths)

    # Infrared region
    ir_indxs = np.where(np.logical_and(0.3 <= x, x < 1.1))
    a[ir_indxs] = 0.574 * x[ir_indxs] ** 1.61
    b[ir_indxs] = -0.527 * x[ir_indxs] ** 1.61

    # Optical and ultraviolet regions
    opt_indxs = np.where(np.logical_and(1.1 <= x, x < 3.3))
    y = x[opt_indxs] - 1.82
    a[opt_indxs] = (1.0 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 +
                    0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 +
                    0.32999 * y**7)
    b[opt_indxs] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 -
                    5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 -
                    2.09002 * y**7)

    # Ultraviolet and NearUV regions till 5.9
    nuv_indxs = np.where(np.logical_and(3.3 <= x, x <= 8.0))
    y = x[nuv_indxs]
    a[nuv_indxs] = 1.752 - 0.316*y - 0.104/((y - 4.67)**2 + 0.341)
    b[nuv_indxs] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62)**2 + 0.263)

    # Near UV region beyond 5.9
    fnuv_indxs = np.where(np.logical_and(5.9 <= x, x <= 8))
    y = x[fnuv_indxs]
    a[fnuv_indxs] += -0.04473*((y - 5.9)**2) - 0.009779*((y - 5.9)**3)
    b[fnuv_indxs] += 0.2130 * ((y - 5.9)**2) + 0.1207*( (y - 5.9)**3)

    # Far UV beyond 8
    fuv_indxs = np.where(np.logical_and(8 < x, x <= 10))
    y = x[fuv_indxs]
    a[fuv_indxs] = -1.073 - 0.628*(y - 8) + 0.137*((y - 8)**2) - 0.070 * ((y - 8)**3)
    b[fuv_indxs] = 13.670 + 4.257*(y - 8) - 0.420*((y - 8)**2) + 0.374*((y-8)**3)

    # Apply extinction correction
    A_lambda = A_v *(a + b / R_v)
    corrected_flux = flux * (10**(A_lambda/2.5))

    return corrected_flux
