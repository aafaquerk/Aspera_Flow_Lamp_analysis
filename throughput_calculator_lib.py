class Reflective_optic:
    def __init__(self, coating, size, radius, name, location):
        self.coating = coating  # dict[float, float]
        self.size = size  # float
        self.radius = radius  # float
        self.name = name  # str
        self.location = location  # int
    
    def set_coating(self, coating: dict[float, float]):
        if isinstance(coating, dict) and len(coating) == 2 and all(isinstance(x, float) for x in coating.values()):
            self.coating = coating
        else:
            raise ValueError("Coating must be a dictionary with two float values.")
    
    def set_size(self, size: float):
        if isinstance(size, float):
            self.size = size
        else:
            raise ValueError("Size must be a float.")
    
    def set_radius(self, radius: float):
        if isinstance(radius, float):
            self.radius = radius
        else:
            raise ValueError("Radius must be a float.")
    
    def set_name(self, name: str):
        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError("Name must be a string.")
    
    def set_location(self, location: int):
        if isinstance(location, int):
            self.location = location
        else:
            raise ValueError("Location must be an integer.")

class Grating_optic():
    def __init__(self, coating, size, radius, name, location,order):
        self.coating = coating  # dict[float, float]
        self.size = size  # float
        self.radius = radius  # float
        self.name = name  # str
        self.location = location  # int
        self.order=order
        self.efficiencies={} #dict[int, float]

    
    def set_coating(self, coating: dict[float, float]):
        if isinstance(coating, dict) and len(coating) == 2 and all(isinstance(x, float) for x in coating.values()):
            self.coating = coating
        else:
            raise ValueError("Coating must be a dictionary with two float values.")
    
    def set_size(self, size: float):
        if isinstance(size, float):
            self.size = size
        else:
            raise ValueError("Size must be a float.")
    
    def set_radius(self, radius: float):
        if isinstance(radius, float):
            self.radius = radius
        else:
            raise ValueError("Radius must be a float.")
    
    def set_name(self, name: str):
        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError("Name must be a string.")
    
    def set_location(self, location: int):
        if isinstance(location, int):
            self.location = location
        else:
            raise ValueError("Location must be an integer.")

    
    def set_order(self, order: int):
        if isinstance(order, int):
            self.order = order
        else:
            raise ValueError("Order must be an integer.")
    
    def set_efficiencies(self, efficiencies: dict[int, float]):
        if isinstance(efficiencies, dict) and all(isinstance(x, int) and isinstance(y, float) for x, y in efficiencies.items()):
            self.efficiencies = efficiencies
        else:
            raise ValueError("Efficiencies must be a dictionary with integer keys and float values.")

class Detector:
    def __init__(self, type, size, pixel_size, masking, wavelength, QE):
        self.type = type  # str
        self.size = size  # tuple
        self.pixel_size = pixel_size  # float
        self.masking = masking  # bool
        self.wavelength = wavelength  # float
        self.QE = QE  # float
    
    def set_type(self, type: str):
        if isinstance(type, str):
            self.type = type
        else:
            raise ValueError("Type must be a string.")
    
    def set_size(self, size: tuple):
        if isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError("Size must be a tuple.")
    
    def set_pixel_size(self, pixel_size: float):
        if isinstance(pixel_size, float):
            self.pixel_size = pixel_size
        else:
            raise ValueError("Pixel size must be a float.")
    
    def set_masking(self, masking: bool):
        if isinstance(masking, bool):
            self.masking = masking
        else:
            raise ValueError("Masking must be a boolean.")
    
    def set_wavelength(self, wavelength: float):
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            raise ValueError("Wavelength must be a float.")
    
    def set_QE(self, QE: float):
        if isinstance(QE, float):
            self.QE = QE
        else:
            raise ValueError("QE must be a float.")

class InputFlux:
    def __init__(self, source_type, wavelength, relative_spectral_irradiance, total_flux, spatial_distribution):
        self.source_type = source_type  # str
        self.wavelength = wavelength  # float
        self.relative_spectral_irradiance = relative_spectral_irradiance  # float
        self.total_flux = total_flux  # float
        self.spatial_distribution = spatial_distribution  # str
    
    def set_source_type(self, source_type: str):
        if isinstance(source_type, str):
            self.source_type = source_type
        else:
            raise ValueError("Source type must be a string.")
    
    def set_wavelength(self, wavelength: float):
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            raise ValueError("Wavelength must be a float.")
    
    def set_relative_spectral_irradiance(self, relative_spectral_irradiance: float):
        if isinstance(relative_spectral_irradiance, float):
            self.relative_spectral_irradiance = relative_spectral_irradiance
        else:
            raise ValueError("Relative spectral irradiance must be a float.")
    
    def set_total_flux(self, total_flux: float):
        if isinstance(total_flux, float):
            self.total_flux = total_flux
        else:
            raise ValueError("Total flux must be a float.")
    
    def set_spatial_distribution(self, spatial_distribution: str):
        if isinstance(spatial_distribution, str):
            self.spatial_distribution = spatial_distribution
        else:
            raise ValueError("Spatial distribution must be a string.")

Input_flux: InputFlux = InputFlux(source_type="Point source", wavelength=0.5, relative_spectral_irradiance=0.5, total_flux=0.5, spatial_distribution="Uniform")
Primary_mirror: Reflective_optic = Reflective_optic(coating={0.5: 0.9, 0.6: 0.8}, size=0.5, radius=0.5, name="Primary mirror", location=0)
Grating: Grating_optic = Grating_optic(coating={0.5: 0.9, 0.6: 0.8}, size=0.5, radius=0.5, name="Grating", location=1, order=1)
