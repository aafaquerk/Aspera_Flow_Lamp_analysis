import plotly.graph_objs as go
app = dash.Dash(__name__)

def rotate_image(data_array, angle):
    rotated_data = rotate(data_array, angle, reshape=False)
    return rotated_data

def process_files_with_dash(filepaths):
    num_files = len(filepaths)
    num_rows = (num_files + 1) // 2  # Calculate the number of rows for subplots

    fig = go.Figure()

    for i, filepath in enumerate(filepaths):
        # Open the FITS file
        hdul = fits.open(filepath)

        # Get the data from the primary HDU
        data = hdul[0].data

        # Convert the data to a NumPy array
        data_array = np.array(data)

        # Close the FITS file
        hdul.close()

        # Rotate the data
        data_rotated = rotate_image(data_array, 37)

        cropped_data = data_rotated[250:1000, :]
        collapsed_spectrum = np.sum(cropped_data, axis=0)
        filtered_spectrum = median_filter(collapsed_spectrum, size=3)

        # Add the filtered spectrum to the figure
        filename = filepath.split('/')[-1]
        fig.add_trace(go.Scatter(x=np.arange(cropped_data.shape[1]), y=filtered_spectrum,
                                 mode='lines', name=filename))

    # Set layout for the figure
    fig.update_layout(
        title="Spectrum (Filtered)",
        xaxis_title="Pixel",
        yaxis_title="Flux (ADU)",
        showlegend=True,
        legend=dict(x=0, y=1),
        grid=dict(visible=False),
        height=500 * num_rows
    )

    # Create the app layout
    app.layout = html.Div(children=[
        html.H1(children='Spectrum Analysis'),

        dcc.Graph(
            id='filtered-spectrum',
            figure=fig
        )
    ])

    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=False)

