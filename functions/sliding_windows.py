import numpy as np


def get_centered_sliding_windows(window_length: int, window_step: int, t_len: int):
    """
    Generate centered sliding windows for time series data.

    Parameters:
    window_length (int): The length of each sliding window.
    window_step (int): The step size for moving the window.
    t_len (int): The total length of the time series data.

    Returns:
    tuple: A tuple containing two elements:
        - A list of arrays, where each array represents the indices of a time window.
        - An array of center points corresponding to the center of each time window.
    """
    # Calculate the half-length of the window to determine start and end points
    half_window_length = window_length // 2
    
    # Generate the center points for each window
    center_points = np.arange(half_window_length, t_len - half_window_length, window_step)
    
    # Initialize a list to store each time window
    time_windows = []
    
    # Create each time window based on the center points
    for center in center_points:
        start = center - half_window_length
        end = center + half_window_length 
        #+ 1  # +1 to include the end point in the window
        # Create a time window from start to end
        time_window = np.arange(start, end)
        time_windows.append(time_window)
    
    return time_windows, center_points