import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

max_speed = 120  # km/h, maximum speed for normalization

def generate_demands():
    """ Generate demand data for simulation based on real traffic speeds.   """
    # Step 1: Load and preprocess the data
    # 1. Load Excel
    df = pd.read_excel(r"C:\Users\anush\Downloads\final_vsl_data.xlsx")[1:]
    df['Mcity_S1'] = df['Mcity_CurrentSpeed'].str.split(',').str[0]
    df['Mcity_S2'] = df['Mcity_CurrentSpeed'].str.split(',').str[1]
    df['Mcity_S3'] = df['Mcity_CurrentSpeed'].str.split(',').str[2]
    # df['Mcity_S1'].unique()
    df = df[df['Mcity_S1'] != 'Error']
    df = df.drop('Mcity_CurrentSpeed', axis=1)

    # Define your observation windows
    specific_day_str = '2025-06-23' # Example: July 16, 2025
    specific_day = pd.to_datetime(specific_day_str).date()
    # morning_window = ('05:00', '09:00')  # 5-9 AM
    evening_window = ('17:30', '20:00')  # 5:30-8 PM

    # Create a mask to keep only data within these windows
    time_condition = (
        # df['Timestamp'].dt.time.between(pd.to_datetime(morning_window[0]).time(),
        #                                pd.to_datetime(morning_window[1]).time()) |
        df['Timestamp'].dt.time.between(pd.to_datetime(evening_window[0]).time(),
                                    pd.to_datetime(evening_window[1]).time())
    )
    combined_condition = time_condition & (df['Timestamp'].dt.date == specific_day)

    print(f"Original rows: {len(df)}")
    # Apply the combined mask to your DataFrame
    df = df[combined_condition].copy()
    print(f"Filtered rows: {len(df)}")

    df = df[['Mcity_S1']].copy()
    speed_data = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    speed_data = speed_data.dropna()  # Remove any NaN values
    # speed_data = speed_data.iloc[:, 0]#.to_numpy()


    # Step 2: Interpolate it to match 900 steps (2.5 hrs @ 10s)
    speed_data.index = np.linspace(0, 2.35, len(speed_data))  # original time (in hours)
    # Interpolate to 900 points (2.5 hrs @ 10 sec)
    upsampled = speed_data.reindex(np.linspace(0, 2.5, 900)).interpolate(method='linear')

    # Convert to numpy array
    speed_array = upsampled.to_numpy()

    # Step 1: Clip speeds to 70–90 for consistent scaling
    clipped = np.clip(speed_array, 70, 90)

    # Step 2: Rescale range [70, 90] → [90, 100]
    # Formula: scaled = (clipped - 70) * (10 / 20) + 90
    demand_values = (clipped - 70) * (10 / 20) + 90

    # Step 5 (optional): Smooth the demand
    demand_values = gaussian_filter1d(demand_values, sigma=2)

    # Step 6: Prepare for MetaNet (combine O1 and O2 demand, same 900 steps)
    d2 = np.ones(900) * 500  # simple constant ramp demand
    demands = np.stack((demand_values, d2), axis=1)  # final shape (900, 2)
    return demands

# # Create 900 time steps (every 10 sec)
# sim_time = np.linspace(0, 2.5, int(2.5 * 3600 / 10))

# # Interpolate
# interp_func = interp1d(real_time, real_speeds, kind='linear')
# sim_speeds = interp_func(sim_time)
