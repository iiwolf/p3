class Variable():

    def __init__(self, name: str, unit: str, add_kilo: bool = True) -> None:
        self.name = name
        self.unit = unit
    
    @property
    def display_str(self):
        if self.unit:
            return f"{self.name} ({self.unit})"
        else:
            return f"{self.name}"

    @property
    def hover_str(self):
        return "%{y:0.2f}" + (f" {self.unit}" if self.unit else "")

