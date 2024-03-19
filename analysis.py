import ipywidgets as widgets
import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import rotate
import pickle
import glob
import os
import re
from scipy.ndimage import median_filter
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
from specutils.fitting import fit_generic_continuum,fit_lines
import specutils
from specutils import Spectrum1D,SpectralRegion,SpectrumCollection
import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models
from astropy import units as u
from astroquery.nist import Nist
import pandas as pd
from astropy.modeling.fitting import LinearLSQFitter
from scipy.signal import find_peaks
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.manipulation import LinearInterpolatedResampler
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_derivative
from specutils.manipulation import noise_region_uncertainty
from specutils.manipulation import gaussian_smooth
from specutils.analysis import line_flux,centroid
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
import scipy.integrate
import pandas as pd
# Example usage
#paths = glob.glob(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/*1223*.fits')
#paths = glob.glob(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/*.fits')
# filter0=''
# filter1= r'/Users/arkhan/Documents/Aspera_Lamp_analysis/*?.fits'
# filter1= r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/*/*/*12**Ar*12*.fits'

# filter2=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/*/*He*.fits'
# filter3=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/*/*12*N2*.fits'
# paths = sorted(glob.glob(filter1,recursive=True)+
#                glob.glob(filter0)+glob.glob(filter0), key=os.path.getmtime)

global angle,image_vmin,image_vmax,ymin,ymax,extract_xmax,extract_xmin,shift_per_step
# Define the extraction region for the spectrum
extract_ymax=900 # max limit in the y direction for extraction_region from the fits image 
extract_ymin=500 # min limit in the y direction for extraction_region from the fits image 
extract_xmax=1560
extract_xmin=640

angle= 33.15# angle of rotation for the fits image to align spectrum with image cartesian coordinates
shift_per_step=18.5 #pixels moved for each step of the grating
# Define the contrast limits for the spectral image plots
image_vmin=0 # contrast limits for the image
image_vmax=15 # contrast limits for the image

# Define the  limits for the spectrum plots
ymin=0 # contrast limits for the spectrum
ymax=1500 # contrast limits for the spectrum


def set_plot_aesthetics():
    plt.style.use({
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
        'figure.figsize': [20, 14],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.size': 7.5,
        'ytick.major.size': 7.5
    })

def find_fits_files(directory):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fits"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def get_spectrum_from_fits(filepath, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax):
    # Get the data from the FITS file
    data_array = fits.getdata(filepath)
    # Calculate the contrast limits based on the data
    data_rotated = rotate_image(data_array, angle)
    cropped_data = data_rotated[extract_ymin:extract_ymax, extract_xmin:extract_xmax]
    spectra_data = u.quantity.Quantity(cropped_data, unit='adu')

    # Convert the cropped data to a Spectrum1D object
    spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(extract_xmin,extract_xmax) * u.pixel)
    # Collapse the spectrum in the spatial direction
    collapsed_spectrum = np.sum(spectrum.flux, axis=0)
    # Apply median filter to the collapsed spectrum
    filtered_spectrum = Spectrum1D(flux=u.quantity.Quantity(median_filter(collapsed_spectrum, size=3)), spectral_axis=spectrum.spectral_axis)
    return filtered_spectrum

def get_spectrum_from_array(spectrum_array):
    """
    Convert a 2D array of spectra data into a filtered spectrum.

    Parameters:
    spectrum_array (ndarray): The input 2D array of spectra data.

    Returns:
    Spectrum1D: The filtered spectrum.

    """
    spectra_data = u.quantity.Quantity(spectrum_array, unit='adu')
    # Convert the cropped data to a Spectrum1D object
    spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(spectra_data.shape[1]) * u.pixel)
    # Collapse the spectrum in the spatial direction
    collapsed_spectrum = np.sum(spectrum.flux, axis=0)
    # Apply median filter to the collapsed spectrum
    filtered_spectrum = Spectrum1D(flux=u.quantity.Quantity(median_filter(collapsed_spectrum, size=3)), spectral_axis=spectrum.spectral_axis)
    return filtered_spectrum

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

def plot_spectrum_1D(spectrum,savepath=None):
            # Create another figure and plot the collapsed spectrum
        fig, ax_spectrum = plt.subplots(figsize=(8, 6))
        ax_spectrum.plot(spectrum.spectral_axis, spectrum.flux)
        ax_spectrum.set_xlabel('Pixel')
        ax_spectrum.set_ylabel('Flux (counts)')
        ax_spectrum.set_title('1D spectrum plot')
        ax_spectrum.set_xlim(extract_xmin, extract_xmax)
        ax_spectrum.set_ylim(ymin, ymax)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        #plt.grid(which='both')
        if savepath!=None: 
            savefile=os.path.dirname(savepath)+'/process_with_file_images.png'
            plt.savefig(savefile)
        # Show the figures
        plt.show()

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
        # Get the filtered spectrum from the FITS file
        filtered_spectrum = get_spectrum_from_fits(filepath, angle, extract_ymin, extract_ymax)

        # Plot the filtered spectrum in the corresponding subplot
        ax = axs[i]
        filename = os.path.basename(filepath)
        ax.plot(filtered_spectrum.spectral_axis, filtered_spectrum.flux, label=os.path.basename(filename))
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Flux (counts)')
        ax.set_title('Spectrum (Filtered) - {}'.format(filename), pad=20)  # Add pad=20 to set_title
        ax.set_xlim(0, 2048)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks
        #ax.grid(which='both')
        ax.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()
    # plt.savefig(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/EUV_test/output/subplots_all.png')
    savefile=os.path.dirname(filepath)+'/subplots_all.png'
    plt.savefig(savefile)

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
        data_rotated = rotate_image(data_array, angle)

        cropped_data = data_rotated[extract_ymin:extract_ymax,:]
        spectra_data = u.quantity.Quantity(cropped_data, unit='adu')

        # Convert the cropped data to a Spectrum1D object
        spectrum = Spectrum1D(flux=spectra_data, spectral_axis=np.arange(cropped_data.shape[1]) * u.pixel)
        # Collapse the spectrum in the spatial direction
        collapsed_spectrum = np.sum(spectrum.flux, axis=0)

        # Plot the cropped data as image in the first column of the subplot
        ax_image = axs[i * 2]
        ax_image.imshow(cropped_data, cmap='gray', vmin=image_vmin, vmax=image_vmax)
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
        ax_spectrum.set_ylim(ymin, ymax)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        #plt.grid(which='both')

    # Adjust the spacing between subplots
    plt.tight_layout()
    savefile=os.path.dirname(filepath)+'/subplots_with_images.png'
    plt.savefig(savefile)

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
        data_rotated = rotate_image(data_array, angle)

        cropped_data = data_rotated[extract_ymin:extract_ymax,:]
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
        ax_image.imshow(cropped_data, cmap='gray', vmin=image_vmin, vmax=image_vmax)
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
        ax_spectrum.set_ylim(ymin, ymax)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        #plt.grid(which='both')
        savefile=os.path.dirname(filepath)+'/process_with_file_images.png'
        plt.savefig(savefile)
        # Show the figures
        plt.show()

def extract_spectral_lines(spectrum, height_threshold, width_threshold):
    # Find the peaks in the spectrum above the threshold
    peaks, _ = find_peaks(spectrum, height=height_threshold)

    # Check if any peaks were found
    if len(peaks) == 0:
        return []

    # Calculate the widths and heights of the peaks
    widths = peak_widths(spectrum, peaks)[0]
    heights = spectrum[peaks]

    # Filter out peaks that are too narrow or too short
    valid_peaks = []
    for i in range(len(peaks)):
        if widths[i] > width_threshold and heights[i] > heights_threshold:
            valid_peaks.append(peaks[i])

    # Extract the spectral lines based on the valid peaks
    spectral_lines = []
    for peak in valid_peaks:
        line = {
            'position': peak,
            'width': widths[peaks.index(peak)],
            'height': heights[peaks.index(peak)]
        }
        spectral_lines.append(line)

    return spectral_lines

def apply_spectral_step_calibration(spectral_axis,step,shift_per_step): 
    # Apply the spectral calibration
    # Define the wavelength calibration parameters
    cal_spectral_axis=((spectral_axis-(min(spectral_axis)+max(spectral_axis))/2)/shift_per_step)+step


    # Calculate the wavelength values

    return cal_spectral_axis

def apply_wavelength_calibration(spectral_axis_step,ref_wavelength,reference_step,cal_wavelenght,cal_step):
    #wave_calibrated=ref_wavelength+((spectral_axis-reference_step)*(cal_wavelenght-ref_wavelength)/(cal_step-reference_step))
    scale=np.abs((cal_wavelenght-ref_wavelength)/(cal_step-reference_step))
    wave_calibrated= (spectral_axis_step-reference_step)*scale+ref_wavelength
    return wave_calibrated

def get_continuum_fit(spectrum): 
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = fit_generic_continuum(spectrum)
    
    y_continuum_fitted = g1_fit(spectrum.spectral_axis)
    fitted_continum=Spectrum1D(flux=y_continuum_fitted, spectral_axis=spectrum.spectral_axis)

def select_file_from_list(filepaths):
    print("Select a file:")
    for i, filepath in enumerate(filepaths):
        print(f"{i+1}. {os.path.basename(filepath)}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file: "))
            if choice < 1 or choice > len(filepaths):
                print("Invalid choice. Please try again.")
            else:
                return filepaths[choice-1]
        except ValueError:
            print("Invalid choice. Please try again.")

def get_image_data(filename):
    data = fits.getdata(filename)
    return data

def fit_spectral_line(spectrum): 
    
    # Fit a Gaussian model to the spectral line
    # Define the initial parameters for the Gaussian model
    amplitude = 1000  # Initial amplitude
    mean =  1100 # Initial mean
    stddev = 0.1  # Initial standard deviation

    # Create the Gaussian model
    gaussian_model = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)

    # Create a fitter object
    fitter = LinearLSQFitter()

    # Fit the model to the spectral line
    fitted_model = fitter(gaussian_model, spectrum.spectral_axis, spectrum.flux)

    # Plot the original spectrum and the fitted model
    plt.plot(spectrum.spectral_axis, spectrum.flux, label='Spectrum')
    plt.plot(spectrum.spectral_axis, fitted_model(spectrum.spectral_axis), label='Fitted Model')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (counts)')
    plt.title('Spectral Line Fitting')
    plt.legend()
    plt.show()

def process_main():
    # process_files_with_subplots(paths)
    # process_files_images(paths)
    # process_files_with_subplots_images(paths)
    fig, ax_spectrum = plt.subplots(figsize=(8, 6))
    minwave = 105 * u.nm
    maxwave = 122 * u.nm
    # then we search for atomic lines
    # We are only interested in neutral lines, assuming the lamps are not hot enough to ionize the atoms
    Ar_lines = Nist.query(minwave,maxwave, linename="Ar I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')
    Ar_lines=Ar_lines[(Ar_lines['Observed']>minwave.value)&(Ar_lines['Observed']<maxwave.value)]


    He_lines = Nist.query(minwave,maxwave, linename="He I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')
    He_lines=He_lines[(He_lines['Observed']>minwave.value)&(He_lines['Observed']<maxwave.value)]

    NI_lines = Nist.query(minwave,maxwave, linename="N I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')

    NI_lines=NI_lines[(NI_lines['Observed']>minwave.value)&(NI_lines['Observed']<maxwave.value)]

    OI_lines = Nist.query(minwave,maxwave, linename="O I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')

    OI_lines=OI_lines[(OI_lines['Observed']>minwave.value)&(OI_lines['Observed']<maxwave.value)]
    print(NI_lines)
    print(Ar_lines)
    print(OI_lines)
    #plt.vlines(Ar_lines['Observed'], 0,250, color='grey', alpha=0.4,label='Ar I')
    #plt.vlines(NI_lines['Observed'], 0,250, color='blue', alpha=0.4,label='N I')
    # plt.vlines(OI_lines['Observed'], 0,250, color='green', alpha=0.4,label='O I')
    # plt.vlines(He_lines['Observed'], 0,250, color='cyan', alpha=0.4,label='He I')
    for filename in paths:
        filtered_spectrum = get_spectrum_from_fits(filename, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
        #plot_spectrum_1D(filtered_spectrum)
        # we adopt the minimum/maximum wavelength from our linear fit
        p=plt.plot(filtered_spectrum.spectral_axis.value, filtered_spectrum.flux,ds='steps-mid',label=os.path.basename(filename))

        #print(filtered_spectrum.flux)
        plt.vlines(filtered_spectrum.spectral_axis.value[np.argmax(filtered_spectrum.flux)], 0,ymax, alpha=0.4,color=p[0].get_color())
        ax_spectrum.set_xlabel('Pixel')
        ax_spectrum.set_ylabel('Flux (counts)')
        ax_spectrum.set_title('1D spectrum plot')
        ax_spectrum.set_xlim(min(filtered_spectrum.spectral_axis.value), max(filtered_spectrum.spectral_axis.value))
        ax_spectrum.set_ylim(ymin, ymax)
        ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

        #plt.grid(which='both')
        # data_array = fits.getdata(paths[0])
        # # Calculate the contrast limits based on the data
        # data_rotated = rotate_image(data_array, angle)[extract_ymin:extract_ymax,750:1500]
        # print(np.sum(data_rotated))
        # background = np.median(data_rotated)
        # yaxis = np.tile(np.arange(extract_ymin,extract_ymax), (750,1))
        # yaxis=np.transpose(yaxis)
        # xvals=np.arange(750)
        # print(yaxis.shape)
        # print(yaxis)
        # # moment 1 is the data-weighted average of the Y-axis coordinates
        # weighted_yaxis=[]
        # for i in range(np.shape(data_rotated)[1]):
        #     weighted_yaxis_values = np.sum(data_rotated[:,i]*yaxis[:,i])/np.sum(data_rotated[:,i])
        #     #print(weighted_yaxis_valuis)
        #     weighted_yaxis.append(weighted_yaxis_values)
        # print(weighted_yaxis_values)
        # median=np.nanmedian(weighted_yaxis)
        # nan_locs=np.where(np.isnan(weighted_yaxis))
        # weighted_yaxis = np.array(weighted_yaxis)    
        # weighted_yaxis[nan_locs]=median
        # # We fit a 2nd-order polynomial
        # polymodel = Polynomial1D(degree=3)
        # linfitter = LinearLSQFitter()
        # fitted_polymodel = linfitter(polymodel,xvals, weighted_yaxis)
        # npixels_to_cut = 170
        # #print(weighted_yaxis)
        # plt.imshow(data_rotated, cmap='viridis', aspect='auto',origin='lower', vmin=0,vmax=15)
        # plt.scatter(xvals,weighted_yaxis-extract_ymin,color='red', s=1)
        # plt.plot(xvals,fitted_polymodel(xvals)-extract_ymin, color='b')
        # plt.colorbar()
        # plt.show()
        # plt.plot(xvals,weighted_yaxis - fitted_polymodel(xvals), 'x')
        # plt.ylabel("Residual (data-model)")
        # plt.show()
        # plt.imshow(data_rotated, cmap='viridis', aspect='auto',origin='lower', vmin=0,vmax=15)
        # plt.fill_between(xvals, fitted_polymodel(xvals)-npixels_to_cut-extract_ymin,
        #                 fitted_polymodel(xvals)+npixels_to_cut-extract_ymin,
        #                 color='orange', alpha=0.5)
        # plt.colorbar()
        # plt.show()
        
        
        # trace_center = fitted_polymodel(xvals)-extract_ymin
        # cutouts = np.array([data_rotated[int(yval)-npixels_to_cut:int(yval)+npixels_to_cut, ii]
        #                     for yval, ii in zip(trace_center, xvals)])
        # spectrum_array=cutouts.T
        # ax1 = plt.subplot(2,1,1)
        # ax1.imshow(data_rotated)
        # ax1.set_title("Oringal Extraction Region")
        # ax2 = plt.subplot(2,1,2)
        # ax2.imshow(spectrum_array)
        # ax2.set_title("Trace-weighted Extraction Region")
        # plt.show()
        # extracted_spectrum=get_spectrum_from_array(spectrum_array)
        # plot_spectrum_1D(extracted_spectrum)
    plt.legend()
    plt.show()

def calc_ar_spectral_shift(): 
    spectrum1_position=1300 #1273
    spectrum2_position=1250
    spectrum1_filter=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/*/*/*1273*Ar*.fits'#1273
    spectrum2_filter=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/*/*/*1250*Ar*12*.fits'
    spectrum1_path = glob.glob(spectrum1_filter,recursive=True)[0]
    spectrum2_path = glob.glob(spectrum2_filter,recursive=True)[0]
    spectrum1 = get_spectrum_from_fits(spectrum1_path, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
    spectrum2 = get_spectrum_from_fits(spectrum2_path, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
    fig,ax_spectrum = plt.subplots(figsize=(8, 6))
    p1=plt.plot(spectrum1.spectral_axis.value, spectrum1.flux,ds='steps-mid',label=os.path.basename(spectrum1_path))
    p2=plt.plot(spectrum2.spectral_axis.value, spectrum2.flux,ds='steps-mid',label=os.path.basename(spectrum2_path))
    #print(filtered_spectrum.flux)
    plt.vlines(spectrum1.spectral_axis.value[np.argmax(spectrum1.flux)], 0,ymax, alpha=0.4,color=p1[0].get_color())
    plt.vlines(spectrum2.spectral_axis.value[np.argmax(spectrum2.flux)], 0,ymax, alpha=0.4,color=p2[0].get_color())
    ax_spectrum.set_xlabel('Pixel')
    ax_spectrum.set_ylabel('Flux (counts)')
    
    ax_spectrum.set_xlim(min(spectrum2.spectral_axis.value), max(spectrum2.spectral_axis.value))
    ax_spectrum.set_ylim(ymin, ymax)
    ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

    lineshift=np.abs(spectrum1.spectral_axis.value[np.argmax(spectrum1.flux)]-spectrum2.spectral_axis.value[np.argmax(spectrum2.flux)])
    step_diff=np.abs(spectrum1_position-spectrum2_position)
    shift_per_step=lineshift/step_diff
    #print(shift_per_step)
    ax_spectrum.set_title(f'1D spectrum plot, Cacluated spectral line shift={shift_per_step} ')
    #plt.grid(which='both')
    plt.show()

def calc_NI_spectral_shift(): 
    spectrum1_position=1000 #1273
    spectrum2_position=1050
    spectrum1_filter=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/03-01-2024/Argon_12psi_0015_HV100mA_HV1.4kV_MCP2.85kV/*1150*.fits'#1273
    spectrum2_filter=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/Nitrogen_15.5psi_0013_HV100mA_HV1.3kV_MCP2.85kV/*1150*.fits'#1273
    spectrum1_path = glob.glob(spectrum1_filter,recursive=True)[0]
    spectrum2_path = glob.glob(spectrum2_filter,recursive=True)[0]
    spectrum1 = get_spectrum_from_fits(spectrum1_path, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
    spectrum2 = get_spectrum_from_fits(spectrum2_path, angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
    fig,ax_spectrum = plt.subplots(figsize=(8, 6))
    p1=plt.plot(spectrum1.spectral_axis.value, spectrum1.flux,ds='steps-mid',label=os.path.basename(spectrum1_path))
    p2=plt.plot(spectrum2.spectral_axis.value, spectrum2.flux,ds='steps-mid',label=os.path.basename(spectrum2_path))
    #print(filtered_spectrum.flux)
    plt.vlines(spectrum1.spectral_axis.value[np.argmax(spectrum1.flux)], 0,ymax, alpha=0.4,color=p1[0].get_color())
    plt.vlines(spectrum2.spectral_axis.value[np.argmax(spectrum2.flux)], 0,ymax, alpha=0.4,color=p2[0].get_color())
    ax_spectrum.set_xlabel('Pixel')
    ax_spectrum.set_ylabel('Flux (counts)')
    
    ax_spectrum.set_xlim(min(spectrum2.spectral_axis.value), max(spectrum2.spectral_axis.value))
    ax_spectrum.set_ylim(ymin, ymax)
    ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major ticks

    lineshift=np.abs(spectrum1.spectral_axis.value[np.argmax(spectrum1.flux)]-spectrum2.spectral_axis.value[np.argmax(spectrum2.flux)])
    step_diff=np.abs(spectrum1_position-spectrum2_position)
    shift_per_step=lineshift/step_diff
    #print(shift_per_step)
    ax_spectrum.set_title(f'1D spectrum plot, ')
    #                    Cacluated spectral line shift={shift_per_step} ')
    plt.legend()
    #plt.grid(which='both')
    plt.show()

def get_nist_lines(Gas_filter, minwave,maxwave):
    ref_lines = Nist.query(minwave,maxwave, linename=f"{Gas_filter} I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')
    min
    ref_lines=ref_lines[(ref_lines['Observed']>minwave.value)&(ref_lines['Observed']<maxwave.value)]
    return ref_lines

def print_nist_lines(minwave,maxwave): 
    minwave = 90 * u.nm
    maxwave = 108 * u.nm
    # then we search for atomic lines
    # We are only interested in neutral lines, assuming the lamps are not hot enough to ionize the atoms
    Ar_lines = Nist.query(minwave,maxwave, linename="Ar I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')
    Ar_lines=Ar_lines[(Ar_lines['Observed']>minwave.value)&(Ar_lines['Observed']<maxwave.value)]


    He_lines = Nist.query(minwave,maxwave, linename="He I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')
    He_lines=He_lines[(He_lines['Observed']>minwave.value)&(He_lines['Observed']<maxwave.value)]

    NI_lines = Nist.query(minwave,maxwave, linename="N I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')

    NI_lines=NI_lines[(NI_lines['Observed']>minwave.value)&(NI_lines['Observed']<maxwave.value)]

    OI_lines = Nist.query(minwave,maxwave, linename="O I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')

    OI_lines=OI_lines[(OI_lines['Observed']>minwave.value)&(OI_lines['Observed']<maxwave.value)]


    HI_lines = Nist.query(minwave,maxwave, linename="H I",
                energy_level_unit='eV', output_order='wavelength',
                wavelength_type='vacuum')

    HI_lines=HI_lines[(HI_lines['Observed']>minwave.value)&(HI_lines['Observed']<maxwave.value)]
    print("HI lines")
    print(HI_lines)
    print("He lines")
    print(He_lines)
    print("Nitrogen lines")
    print(NI_lines)
    print("Oxygen lines")
    print(OI_lines)
    print("Argon lines")
    print(Ar_lines)
    return 

def generate_fits_database(path=r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data'):
    allfiles=find_fits_files(path)
    #print(allfiles)
    # Create a list to store the file names and exposure times
    file_list = []
    step_list= []
    filepaths=[]
    # Iterate over the file paths
    for idx,filepath in enumerate(allfiles):
        # Get the file name
        filename = os.path.basename(filepath)
        
        # Read the fits header
        header = fits.getheader(filepath)
        
        # Get the exposure time from the header
        exposure_time = header['INT-TIME']
        
        # Append the file name and exposure time to the list if the filename has a step number
        if extract_numbers_from_filename(filename,4) is not None:
            file_list.append((filename, exposure_time))
            step_list.append(extract_numbers_from_filename(filename,4))
            filepaths.append(filepath)
        else: 
            allfiles=allfiles.pop(idx)
        

    # Create a pandas DataFrame from the file list
    df = pd.DataFrame( file_list, columns=['File Name', 'Exposure Time'])
    df['File Path']=filepaths
    df['Step'] = step_list
    df['Gas Type'] = df['File Name'].str.extract(r'_([A-Z][a-z0-9])_')
    df.to_csv(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/fits_database.csv',index=False)
    return df

def extract_numbers_from_filename(filename,consecutive_numbers=4):
    string=f"\d"+"{"+f"{consecutive_numbers}"+"}"
    numbers = re.findall(string, filename)
    if len(numbers)==0:
        return None
    else:
        return [int(num) for num in numbers][0]


if __name__ == "__main__":
    set_plot_aesthetics()
    df=generate_fits_database()
    df=pd.read_csv(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/fits_database.csv')
    df.sort_values(by=['Step'],inplace=True)
    df.to_html(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/fits_database.html')
    fig,ax_spectrum = plt.subplots(figsize=(15, 9))
    ymax_lim=0
    first_idx=0
    Gas_filter='Ar'
    file_filter='03_01'
    file_filter2='_'
    exp_filter='120sec'
    try: 
        for idx, row in df.iterrows():
            filter=(Gas_filter in row['File Path'])&((file_filter in row['File Path'])&(file_filter2 in row['File Path']))&(exp_filter in row['File Path'])#|('He' in row['File Path'])) #&('1253' in row['File Path'])
            if filter:
                filename=row['File Name']
                step=row['Step']
                filtered_spectrum = get_spectrum_from_fits(row['File Path'], angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
                callambda=apply_spectral_step_calibration(filtered_spectrum.spectral_axis.value,step,shift_per_step)
                #print(callambda)
                calspectrum=Spectrum1D(filtered_spectrum.flux/(row['Exposure Time']),callambda*u.AA)
                callambda=apply_wavelength_calibration(calspectrum.spectral_axis.value,ref_wavelength=1048.22,reference_step=1157.97,cal_wavelenght=1066.66,cal_step=1178.51)
                binsize=(callambda[1]-callambda[0])
                calflux=calspectrum.flux#pixelscale 
                calspectrum=Spectrum1D(calspectrum.flux/binsize,callambda*u.AA)
                if first_idx==0: 
                    full_specturm=calspectrum
                    first_idx=1
                else: 
                    new_spectral_axis = np.concatenate([full_specturm.spectral_axis.value, calspectrum.spectral_axis.to_value(full_specturm.spectral_axis.unit)]) * full_specturm.spectral_axis.unit
                    resampler = LinearInterpolatedResampler(extrapolation_treatment='zero_fill')
                    new_fullspectrum = resampler(full_specturm, new_spectral_axis)
                    new_calspecturm = resampler(calspectrum, new_spectral_axis)
                    full_specturm=new_fullspectrum+new_calspecturm
                    #print(full_specturm)
                # p=plt.plot(calspectrum.spectral_axis.value, calspectrum.flux,
                #         #    label=filename,
                #            alpha=0.5,color='blue')
                # ax_spectrum.set_xlabel('Step')
                # ax_spectrum.set_ylabel('Flux (counts/s)')
                # ymax_lim=max(calspectrum.flux) if max(calspectrum.flux)>ymax_lim else ymax_lim
                # ax_spectrum.set_xlim(1000,1600)
                # ax_spectrum.set_ylim(ymin, ymax_lim/row['Exposure Time'])
                # ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='out')  # decorating minor and major tic
        # print(full_specturm)
        # fig,ax_spectrum = plt.subplots(figsize=(18, 16))
        p=plt.plot(full_specturm.spectral_axis.value, full_specturm.flux,label=f'{Gas_filter} data',alpha=0.8,linewidth=0.6,color='firebrick',ds='steps-mid')
        ax_spectrum.set_xlabel(rf'$Wavelength\; (\AA)$',labelpad=8)
        ax_spectrum.set_ylabel(rf'$Flux\; (counts/s/\AA)$',labelpad=8)
        ymax_lim=max(full_specturm.flux) if max(full_specturm.flux)>ymax_lim else ymax_lim
        #ax_spectrum.set_xlim(1000,1600)
        ax_spectrum.set_ylim(ymin,ymax_lim)
        # ax_spectrum.tick_params(which='both', width=1.5, length=3.5, direction='in')  # decorating minor and major tic        
        ax_spectrum.set_title(f"Spectra for {Gas_filter} between steps 1000,1600 \n Filters = {file_filter} & {Gas_filter}")
        #plt.grid(which='both')
        #print(full_specturm.spectral_axis)
    except KeyboardInterrupt: 
        print("Interrupted")
    full_specturm_raw=full_specturm
    full_specturm = gaussian_smooth(full_specturm, stddev=10)
    # Define a noise region for adding the uncertainty
    noise_region = SpectralRegion(1000*u.AA, 1100*u.AA)
    spectrum = noise_region_uncertainty(full_specturm, noise_region)
    lines = find_lines_derivative(full_specturm, flux_threshold=25)  
    lines[lines['line_type'] == 'emission']  
    df_lines=lines.to_pandas()
    df_lines.to_csv(r'/Users/arkhan/Documents/Aspera_Lamp_analysis/data/NI_lines_from_spectra.csv',index=False)
    # for idx, rows in df_lines.iterrows(): 
    #     # print(rows.line_center)
    #     plt.vlines(rows.line_center, 
    #             ymin=0,
    #             ymax=ymax_lim, 
    #             alpha=0.4,
    #             label=rf"line center @ {rows.line_center:.2f} $\AA$",
    #             color='black',
    #             linestyle='dotted')

    # peaks,_= find_peaks(full_specturm.flux, distance=20)
    # plt.plot(peaks, full_specturm.flux[peaks], "o")
    # plt.vlines(peaks, ymin=0,ymax=ymax_lim, alpha=0.4)
    # print(np.shape(peaks))

    linecetners = [rows.line_center for idx, rows in df_lines.iterrows()]
    window=5
    subregions = [SpectralRegion(cw*u.AA-window*u.AA, cw*u.AA+window*u.AA) for cw in linecetners]
    sub_spectrums = [extract_region(full_specturm, subregions[i]) for i in range(len(subregions))]
    gaussian_data = pd.DataFrame(columns=['Amplitude', 'Mean', 'Stddev'])
    for sub_spectrum in sub_spectrums:
        try: 
            amp = np.argmax(sub_spectrum.flux)
            peak = sub_spectrum.spectral_axis[np.argmax(sub_spectrum.flux)]
            # Fit the spectrum
            g1_init = models.Gaussian1D(amplitude=amp, mean=peak, stddev=0.5*u.AA)
            g1_fit = fit_lines(sub_spectrum, g1_init, window=1*u.AA)
            y1_fit = g1_fit(sub_spectrum.spectral_axis)
            plt.plot(sub_spectrum.spectral_axis, y1_fit, label=fr'Gaussian @ {sub_spectrum.spectral_axis.value[np.argmax(sub_spectrum.flux)]:.2f} $\AA$')
            
            # Append Gaussian properties to DataFrame
            gaussian_data = gaussian_data.append({'Amplitude': g1_fit.amplitude.value,
                                                  'Mean': g1_fit.mean.value,
                                                  'Stddev': g1_fit.stddev.value}, ignore_index=True)
        except: 
            print("No fit for subregion")

     
    gaussians_filename= os.path.join('/Users/arkhan/Documents/Aspera_Lamp_analysis/data/'+f'{Gas_filter}_lines_from_spectra_gaussian.csv')
    gaussian_data.to_csv(gaussians_filename,index=False)
    ref_lines=get_nist_lines(Gas_filter,min(full_specturm.spectral_axis),max(full_specturm.spectral_axis))
    NI_lines=get_nist_lines('N2',min(full_specturm.spectral_axis),max(full_specturm.spectral_axis))
    OI_lines=get_nist_lines('O',min(full_specturm.spectral_axis),max(full_specturm.spectral_axis))
    HI_lines=get_nist_lines('H',min(full_specturm.spectral_axis),max(full_specturm.spectral_axis))
    plt.minorticks_on()
    # Add secondary x-axis
    plt.tight_layout()
    fig_file=f"Wavelength_calibrated_{Gas_filter}_spectra_with_lines.png"
    plt.savefig(fig_file)
    pickle.dump((fig, ax_spectrum), open(f'{fig_file[:-4]}.pickle', 'wb'))
    plt.show()
    argon_1048_region= SpectralRegion(1042*u.AA,1054*u.AA)
    argon_1048_specturm=extract_region(full_specturm,argon_1048_region)
    argon_1048_sigma_width=gaussian_sigma_width(argon_1048_specturm)
    print(f'{argon_1048_sigma_width =}')
    argon_1048_centroid=centroid(argon_1048_specturm)
    print(f'{argon_1048_centroid =}')
    argon_1048_region_5sigma= SpectralRegion(argon_1048_centroid-2.5*argon_1048_sigma_width,argon_1048_centroid+2.5*argon_1048_sigma_width)
    argon_1048_specturm_5sigma=extract_region(full_specturm,argon_1048_region_5sigma)
    bins=len(argon_1048_specturm_5sigma.spectral_axis)
    print(f'{bins =}')
    # bin_size=(argon_1048_specturm_5sigma.spectral_axis[1]-argon_1048_specturm_5sigma.spectral_axis[0])
    # print(bin_size)
    argon_1048_lineflux=scipy.integrate.trapz(argon_1048_specturm_5sigma.flux.value,argon_1048_specturm_5sigma.spectral_axis.value)
    print(f'{argon_1048_lineflux =}')
    plt.plot(argon_1048_specturm_5sigma.spectral_axis.value, argon_1048_specturm_5sigma.flux.value, label='Argon 1048')
    plt.scatter(argon_1048_specturm_5sigma.spectral_axis.value, argon_1048_specturm_5sigma.flux.value)
    plt.vlines(argon_1048_centroid.value, 0, max(argon_1048_specturm_5sigma.flux.value), alpha=0.4, color='black')
    plt.vlines(argon_1048_centroid.value-2.5*argon_1048_sigma_width.value, 0, max(argon_1048_specturm_5sigma.flux.value), alpha=0.4, color='black', linestyles='dotted')
    plt.vlines(argon_1048_centroid.value+2.5*argon_1048_sigma_width.value, 0, max(argon_1048_specturm_5sigma.flux.value), alpha=0.4, color='black', linestyles='dotted')
    plt.xlabel(rf'$Wavelength (\AA)$', labelpad=8)
    plt.ylabel('$Flux (counts/s/\AA)$', labelpad=8)
    plt.title('Argon 1048 Spectrum')
    plt.gca().tick_params(axis='both', pad=5)
    plt.legend()
    plt.xlim(1042, 1054)
    plt.ylim(0, max(argon_1048_specturm.flux.value))
    plt.show()
