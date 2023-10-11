from p3.core.variable import Variable

def get_display_str(var):
    if var in variable_labels:
        return variable_labels[var].display_str
    else:
        return var
    
def get_hover_str(var):
    if var in variable_labels:
        return variable_labels[var].hover_str
    else:
        return var
    
variable_labels = {
    "t" : Variable("Time", "s"),
    "stage" : Variable("Stage", None),
    "ax" : Variable("Horizontal Acceleration", "m/s^2"),
    "ay" : Variable("Vertical Acceleration", "m/s^2"),
    "fx" : Variable("Horizontal Force", "N"),
    "fy" : Variable("Vertical Force", "N"),
    "g" : Variable("Gravity", "m/s^2"),
    "rho" : Variable("Density", "kg/m^3"),
    "x" : Variable("Ground Range", "m"),
    "y" : Variable("Altitude", "m"),
    "vx" : Variable("Horizontal Velocity", "m/s"),
    "vy" : Variable("Vertical Velocity", "m/s"),
    "speed" : Variable("Speed", "m/s"),
    "mach" : Variable("Mach", None),
    "mass" : Variable("Mass", "kg"),
    "fuel_mass" : Variable("Fuel Mass", "kg"),
    "drag" : Variable("Drag", "N"),
    "lift" : Variable("Lift", "N"),
    "cd" : Variable("Drag Coeff.", None),
    "cl" : Variable("Lift Coeff.", None),
    "thrust" : Variable("Thrust", "N"),
    "alpha" : Variable("Alpha", "deg"),
    "gamma" : Variable("Gamma", "deg"),
}

FLIGHT_DASH_VARS = [
    "stage", 
    "alpha", "gamma",
    "x", "y", 
    "vx", "vy",
    "ax", "ay", 
    "fx", "fy", 
    "drag", "lift", "thrust",
    "cd", "cl", 
    # "rho", @TODO: should be included I guess
    "speed", "mach",
    "mass", "fuel_mass",  
]