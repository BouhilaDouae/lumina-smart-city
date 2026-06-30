import os
import pandas as pd
from prediction.graph_builder import clean_load_excel

def load_traffic_data(data_dir="data"):
    """Loads and cleans the 30-day traffic dataset."""
    traffic_file = os.path.join(data_dir, "17_Traffic_30min_30days.xlsx")
    df = clean_load_excel(traffic_file)
    
    # Ensure proper data types
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Pedestrians'] = pd.to_numeric(df['Pedestrians'])
    df['Vehicles'] = pd.to_numeric(df['Vehicles'])
    df['Cyclists'] = pd.to_numeric(df['Cyclists'])
    
    # Sort chronologically and by Node_ID
    df = df.sort_values(by=['Timestamp', 'Node_ID']).reset_index(drop=True)
    return df

def load_weather_data(data_dir="data"):
    """Loads and cleans the hourly weather dataset."""
    weather_file = os.path.join(data_dir, "22_Weather_30days.xlsx")
    df = clean_load_excel(weather_file)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in ['Temp_C', 'Feels_Like_C', 'Humidity_pct', 'Wind_Speed_kmh', 'Pressure_hPa', 'Rain_mm', 'UV_Index', 'Visibility_km', 'Recommended_Brightness_Boost_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Forward-fill any minor missing weather records
    df = df.sort_values(by='Timestamp').ffill().reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Loading traffic and weather data...")
    df_traffic = load_traffic_data()
    df_weather = load_weather_data()
    print(f"Traffic data loaded: {df_traffic.shape[0]} rows, columns: {df_traffic.columns.tolist()}")
    print(f"Weather data loaded: {df_weather.shape[0]} rows, columns: {df_weather.columns.tolist()}")
