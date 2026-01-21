from pathlib import Path
import scipy
import numpy as np

from sim import PROJECT_HOME

def get_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    else:
        return str(Path(PROJECT_HOME, path))
    
def interpolate_loggp_params(params: dict) -> dict:
    """
    Interpolates the LogGP parameters for a given size.

    Args:
        params (dict): Dictionary of LogGP parameters.

    Returns:
        dict: Interpolated LogGP parameters.
    """

    interp_params = {}
    for file_name, params in params.items():
        param_names = ["L","o_s","o_r","g","G"]
        interp_param = {}
        # Interpolation function either interpolates the parameter or if the data is outisde
        # the range of the data, it will use the value associated with the largest size
        for name in param_names:
            sizes = [s for s in params.keys()]
            values = [params[s][name] for s in sizes]
            max_value_idx = np.argmax(sizes)
            interp_function = scipy.interpolate.interp1d(sizes, values, kind='linear', 
                                                         fill_value=np.array([values[max_value_idx]]), 
                                                         bounds_error=False)
            interp_param[name] = interp_function

        interp_params[file_name] = interp_param

            
    return interp_params