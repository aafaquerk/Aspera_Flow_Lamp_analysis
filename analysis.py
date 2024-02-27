import ipywidgets as widgets
import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import rotate
import specutils
from specutils import Spectrum1D
import glob
import os
from scipy.ndimage import median_filter
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.ndimage import median_filter
from specutils.fitting import fit_generic_continuum
import matplotlib.pyplot as plt
import warnings

# Setting plot aesthetics
plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rcParams['figure.figsize'] = [20, 14]  # set plot size


def process_files_with_subplots(filepaths):
    """
    Process FITS files and plot the filtered spectrum in subplots.

    Args:
        filepaths (list): List of file paths to FITS files.

    Returns:
        None
    """
    num_files = len(filepaths)
    num_rows = (num_files + 1) // 2  # Calculate the number of rows for subplots

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    axs = axs.flatten()  # Flatten the 2D array of subplots

    for i, filepath in enumerate(filepaths):
        # Open the FITS file
        hdul = fits.open(filepath)

        # Get the data from the primary HDU
        data = hdul[0].data

        # Convert the data to a NumPy array
        data_array = np.array(data)

        # Close the FITS file
        hdul.close()

        def rotate_image(data_array, angle):
            """
            Rotate the input data array by the specified angle.

            Args:
                data_array (ndarray): Input data array.
                angle (float): Rotation angle in degrees.

            Returns:
                ndarray: Rotated data array.
            """
            rotated_data = rotate(data_array, angle, reshape=False)
            return rotated_data

        # Calculate the contrast limits based on the data
        data_rotated = rotate_image(data_array, 37)

        cropped_data = data_rotated[250:1000, :]
        spectra_data = u.quantity.Quantity(cropped_data, unit='adu')

        # Convert the cropped data to a Spectrum1D object
        spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(cropped_data.shape[1]) * u.pixel)
        # Collapse the spectrum in the spatial direction
        collapsed_spectrum = np.sum(spectrum.flux, axis=0)

        # Apply median filter to the collapsed spectrum
        filtered_spectrum = median_filter(collapsed_spectrum, size=3)

        # Plot the filtered spectrum in the corresponding subplot
        ax = axs[i]
        filename = os.path.basename(filepath)
        ax.plot(spectrum.spectral_axis, filtered_spectrum, label=os.path.basename(filename))
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Flux (counts)')
        ax.set_title('Spectrum (Filtered) - {}'.format(filename), pad=20)  # Add pad=20 to set_title
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2500)
        ax.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks
        ax.grid(which='both')
        ax.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure with subplots
    plt.show()


def process_files_with_subplots_images(filepaths):
    """
    Process FITS files and plot the cropped data as image and collapsed spectrum in subplots.

    Args:
        filepaths (list): List of file paths to FITS files.

    Returns:
        None
    """
    num_files = len(filepaths)
    num_rows = (num_files + 1)  # Calculate the number of rows for subplots

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    axs = axs.flatten()  # Flatten the 2D array of subplots

    for i, filepath in enumerate(filepaths):
        # Open the FITS file
        hdul = fits.open(filepath)

        # Get the data from the primary HDU
        data = hdul[0].data

        # Convert the data to a NumPy array
        data_array = np.array(data)

        # Close the FITS file
        hdul.close()

        def rotate_image(data_array, angle):
            """
            Rotate the input data array by the specified angle.

            Args:
                data_array (ndarray): Input data array.
                angle (float): Rotation angle in degrees.

            Returns:
                ndarray: Rotated data array.
            """
            rotated_data = rotate(data_array, angle, reshape=False)
            return rotated_data

        # Calculate the contrast limits based on the data
        data_rotated = rotate_image(data_array, 37)

        cropped_data = data_rotated[250:1000, :]
        spectra_data = u.quantity.Quantity(cropped_data, unit='adu')

        # Convert the cropped data to a Spectrum1D object
        spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(cropped_data.shape[1]) * u.pixel)
        # Collapse the spectrum in the spatial direction
        collapsed_spectrum = np.sum(spectrum.flux, axis=0)

        # Plot the cropped data as image in the first column of the subplot
        ax_image = axs[i * 2]
        ax_image.imshow(cropped_data, cmap='gray', vmin=0, vmax=15)
        ax_image.set_xlabel('Pixel')
        ax_image.set_ylabel('Line')
        ax_image.set_title('Cropped Data - {}'.format(os.path.basename(filepath)))
        ax_image.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        # Plot the collapsed spectrum in the second column of the subplot
        ax_spectrum = axs[i * 2 + 1]
        ax_spectrum.plot(spectrum.spectral_axis, collapsed_spectrum)
        ax_spectrum.set_xlabel('Pixel')
        ax_spectrum.set_ylabel('Flux (counts)')
        ax_spectrum.set_title('Collapsed Spectrum - {}'.format(os.path.basename(filepath)))
        ax_spectrum.set_xlim(0, 2048)
        ax_spectrum.set_ylim(0, 2500)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        plt.grid(which='both')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/output/plots_with_images.png')

    # Show the figure with subplots
    plt.show()


def process_files_images(filepaths):
    """
    Process FITS files and plot the cropped data as image and collapsed spectrum in separate figures.

    Args:
        filepaths (list): List of file paths to FITS files.

    Returns:
        None
    """
    for filepath in filepaths:
        # Open the FITS file
        hdul = fits.open(filepath)

        # Get the data from the primary HDU
        data = hdul[0].data

        # Convert the data to a NumPy array
        data_array = np.array(data)

        # Close the FITS file
        hdul.close()

        def rotate_image(data_array, angle):
            """
            Rotate the input data array by the specified angle.

            Args:
                data_array (ndarray): Input data array.
                angle (float): Rotation angle in degrees.

            Returns:
                ndarray: Rotated data array.
            """
            rotated_data = rotate(data_array, angle, reshape=False)
            return rotated_data

        # Calculate the contrast limits based on the data
        data_rotated = rotate_image(data_array, 37)

        cropped_data = data_rotated[250:1000, :]
        spectra_data = u.quantity.Quantity(cropped_data, unit='adu')

        # Convert the cropped data to a Spectrum1D object
        spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(cropped_data.shape[1]) * u.pixel)
        # Collapse the spectrum in the spatial direction
        collapsed_spectrum = np.sum(spectrum.flux, axis=0)
        # Apply median filter to the collapsed spectrum
        filtered_spectrum = Spectrum1D(flux=u.quantity.Quantity(median_filter(collapsed_spectrum, size=3)), spectral_axis=spectrum.spectral_axis)

        # Fit and remove continuum in the spectrum
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            g1_fit = fit_generic_continuum(filtered_spectrum)

        # Create a new figure and plot the cropped data as image
        fig, ax_image = plt.subplots(figsize=(8, 6))
        ax_image.imshow(cropped_data, cmap='gray', vmin=0, vmax=15)
        ax_image.set_xlabel('Pixel')
        ax_image.set_ylabel('Line')
        ax_image.set_title('Cropped Data - {}'.format(os.path.basename(filepath)))
        ax_image.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        plt.show()

        # Create another figure and plot the collapsed spectrum
        fig, ax_spectrum = plt.subplots(figsize=(8, 6))
        ax_spectrum.plot(filtered_spectrum.spectral_axis, filtered_spectrum.flux)
        ax_spectrum.set_xlabel('Pixel')
        ax_spectrum.set_ylabel('Flux (counts)')
        ax_spectrum.set_title('Collapsed Spectrum - {}'.format(os.path.basename(filepath)))
        ax_spectrum.set_xlim(0, 2048)
        ax_spectrum.set_ylim(0, 2500)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        plt.grid(which='both')

        # Show the figures
        plt.show()


# Example usage
paths = [r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/test_12_19_1223_He_30sec.fits']
paths = glob.glob(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/*.fits')

process_files_with_subplots(paths)
process_files_images(paths)