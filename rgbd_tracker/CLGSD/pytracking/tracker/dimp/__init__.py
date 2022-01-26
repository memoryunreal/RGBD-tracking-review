from .dimp import DiMP
from .mydimp import DiMP as myDiMP

def get_tracker_class():
    return DiMP

def get_my_tracker_class():
    return myDiMP
