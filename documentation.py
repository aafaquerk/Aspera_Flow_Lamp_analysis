import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import h, c
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.visualization import quantity_support

bb = BlackBody(temperature=298*u.K)
wav = (np.arange(200, 1000) * u.nm).to(u.m)
spectral_radiance = bb(wav).si
solid_angle=2*np.pi*u.sr
detector_area=((13.5*u.micron)**2).to(u.m**2)

spectral_irradiance = (spectral_radiance * solid_angle)
radiant_flux=(spectral_irradiance*detector_area).to(u.W*u.s)
flux_photon = (radiant_flux/(h * c /wav))
with quantity_support():
    plt.figure()
    plt.semilogy(wav, flux_photon)
    plt.show()