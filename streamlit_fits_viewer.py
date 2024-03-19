import streamlit as st
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import rotate
# import specutils
# Define the extraction region for the spectrum
global extract_ymax, extract_ymin, extract_xmax, extract_xmin, angle

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

def main():
    # Define the extraction region for the spectrum
    extract_ymin: int=550 
    extract_ymax: int=900 
    extract_xmin: int=600
    extract_xmax: int=1600
    
    angle: float= 33.15# angle of rotation for the fits image to align spectrum with image cartesian coordinates

    st.title("FITS File Viewer")

    uploaded_file = st.file_uploader("Upload FITS file", type=["fits"])

    if uploaded_file is not None:
        with fits.open(uploaded_file) as hdulist:
            st.sidebar.header("FITS Header")

            st.sidebar.text(hdulist[0].header.tostring(sep='\n'))

            st.subheader("Image")
            image_data = hdulist[0].data
            # filtered_spectrum = get_spectrum_from_fits(row['File Path'], angle, extract_ymin, extract_ymax,extract_xmin,extract_xmax)
                        # Add a textbox for the rotation angle
            # angle = st.text_input('Enter the rotation angle', value=str(angle), key='rotation_angle')

            # Add checkboxes for the rotation and cropping options
            rotate_checkbox = st.checkbox('Rotate Image')
            crop_checkbox = st.checkbox('Crop Image')

            # Initialize variables for rotation and cropping
            new_angle = float(angle)
            new_extract_ymin = int(extract_ymin)
            new_extract_ymax = int(extract_ymax)
            new_extract_xmin = int(extract_xmin)
            new_extract_xmax = int(extract_xmax)

            # Check if rotation checkbox is enabled
            if rotate_checkbox:
                # Add a text input for the rotation angle
                new_angle = st.number_input('Enter the rotation angle', value=float(angle), min_value=0.00, max_value=360.00-0.01, format="%.2f", step=0.01, key='rotation_angle')
                new_angle=float(new_angle)
            else: 
                new_angle = angle
            # Check if cropping checkbox is enabled
            if crop_checkbox:
                # Add text inputs for the extraction region extents
                new_extract_ymin = st.number_input('Enter the min y value for extraction region', value=int(extract_ymin), min_value=0, max_value=image_data.shape[0], step=1, key='extract_ymin')
                new_extract_ymax = st.number_input('Enter the max y value for extraction region', value=int(extract_ymax), min_value=0, max_value=image_data.shape[0], step=1, key='extract_ymax')
                new_extract_xmin = st.number_input('Enter the min x value for extraction region', value=int(extract_xmin), min_value=0, max_value=image_data.shape[1], step=1, key='extract_xmin')
                new_extract_xmax = st.number_input('Enter the max x value for extraction region', value=int(extract_xmax), min_value=0, max_value=image_data.shape[1], step=1, key='extract_xmax')
            else:
                new_extract_ymin = extract_ymin
                new_extract_ymax = extract_ymax
                new_extract_xmin = extract_xmin
                new_extract_xmax = extract_xmax
            # Add an update button
            if st.button('Update'):
                # Check if angle is not empty and is a number
                if new_angle != "" and new_angle is not None:
                    angle = float(new_angle)
                    # Rotate the image here
                    image_data = rotate_image(image_data, angle)
                else:
                    st.warning("Please enter a valid angle.")

                # Check if the extents are valid numbers
                if new_extract_ymin != "" and new_extract_ymax != "" and new_extract_xmin != "" and new_extract_xmax != "" and new_extract_ymin is not None and new_extract_ymax is not None and new_extract_xmin is not None and new_extract_xmax is not None:
                    # Convert the extents to integers
                    extract_ymin = int(new_extract_ymin)
                    extract_ymax = int(new_extract_ymax)
                    extract_xmin = int(new_extract_xmin)
                    extract_xmax = int(new_extract_xmax)

                    # Check if min is less than max
                    if extract_ymin >= extract_ymax or extract_xmin >= extract_xmax:
                        st.warning("Please make sure the min value is less than the max value for extraction region.")
                    # Crop the image here
                    image_data = image_data[extract_ymin:extract_ymax, extract_xmin:extract_xmax]
                else:
                    st.warning("Please enter valid extents.")
            if isinstance(image_data, np.ndarray):
                min_value = np.min(image_data)
                if min_value < 0:
                    min_value = 0
                max_value = 20

                min_max_values = st.slider("Min/Max Values", min_value, max_value, (min_value, max_value))
                # fig = go.Figure(data=go.Heatmap(z=image_data, colorscale="gray", zmin=min_max_values[0], zmax=min_max_values[1]))
                fig = px.imshow(image_data, aspect='equal', zmin=min_max_values[0], zmax=min_max_values[1],origin='lower')
                fig.update_layout(width=720, height=720, margin=dict(l=10, r=10, b=20, t=20, pad=4))
                fig['layout']['yaxis'].update(autorange = True)
                fig.update_layout(
                autosize=True,
                width=720,
                margin=dict(
                    l=10,
                    r=10,
                    b=20,
                    t=20,
                    pad=4
                )
                )

                st.plotly_chart(fig, use_container_width=False,theme="streamlit")


                st.subheader("1D Spectrum")


                # # spectrum = get_spectrum_from_fits(image_data,extract_xmax=extract_xmax,extract_xmin=extract_xmin,extract_ymin=extract_ymin,extract_ymax=extract_ymax)
                # # fig_spectrum = go.Figure(data=go.Scatter(x=np.arange(len(spectrum)), y=spectrum))
                # fig_spectrum.update_layout(width=720, height=360)
                # st.plotly_chart(fig_spectrum, use_container_width=False, theme="streamlit")
            else:
                st.warning("No image data found in the FITS file.")

if __name__ == "__main__":
    main()
