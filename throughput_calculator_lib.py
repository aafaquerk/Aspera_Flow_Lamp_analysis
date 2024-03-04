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
