import pandas as pd
import numpy as np

COLUMN_MAP = {
    "time": "Time",
    "baroaltitude": "Altitude",
    "vertratecorr": "VerticalSpeed",
    "PTCH": "Pitch",
    "ROLL": "Roll",
    "HDGS": "Yaw",
    "TAS": "Speed",
    "LATP": "Latitude",
    "LONP": "Longitude"
}

REQ_COLS = ['Time', 'Longitude', 'Latitude', 'Altitude',
            'Roll', 'Pitch', 'Yaw']



def load_csv(path):
    """The flight data is loaded with the required columns only"""

    df = pd.read_csv(
        path,
        usecols=lambda col: col in list(COLUMN_MAP.keys()) + [
            "GMT_HOUR", "GMT_MINUTE", "GMT_SEC"
        ]
    )

    # --- Step 2: CREATE TIME (ADD HERE) ---
    df["Time"] = (
        df["GMT_HOUR"] * 3600 +
        df["GMT_MINUTE"] * 60 +
        df["GMT_SEC"]
    )
            
    df = df.rename(columns=COLUMN_MAP)
    
    missing = set(REQ_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=REQ_COLS)
    df = df.sort_values("Time").reset_index(drop=True)
            
    return df[REQ_COLS]



def normalize(df):
    """ Clean the data and sort it by time in ascending """

    df = df.apply(pd.to_numeric, errors='coerce')
    
    return (df.dropna()
             .sort_values("Time")
             .drop_duplicates("Time")
             .reset_index(drop=True))



def interpolate(df, rate_hz=30):
    """ Resample the values so that the time intervals are uniform  """

    if len(df) < 2:
        raise ValueError("Need at least 2 data points to interpolate.")
    
    start, end = df["Time"].iloc[[0, -1]]
    new_time = np.arange(start, end, 1.0 / rate_hz)
    
    old_time = df["Time"].values
    
    result = {"Time": new_time}
    for col in df.columns:
        if col != "Time":
            result[col] = np.interp(new_time, old_time, df[col].values)
    
    return pd.DataFrame(result)


def preprocess_flight_data(input_path, output_path, rate_hz=30):
    """ Preprocessing pipeline  """

    df = load_csv(input_path)
    df = normalize(df)
    df = interpolate(df, rate_hz)
    df.to_csv(output_path, index=False)
    
    print(f" {len(df)} data points were processed at {rate_hz} Hz")
    print(f" The values were saved to: {output_path}")
    
    return output_path
