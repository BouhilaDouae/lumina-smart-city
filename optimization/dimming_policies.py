import numpy as np

def static_baseline_policy(row):
    """
    Static Baseline Policy: Streetlights operate at 100% constant brightness 
    during all active night hours.
    """
    return 100.0

def rule_based_control_policy(row, ped_col='Pedestrians', veh_col='Vehicles'):
    """
    Rule-Based Control (RBC) Policy:
    Maps actual/predicted traffic to fixed discrete dimming levels:
      - High Traffic (Pedestrians > 30 or Vehicles > 15) -> 100% brightness.
      - Medium Traffic (Pedestrians > 10 or Vehicles > 5) -> 60% brightness.
      - Low Traffic (others) -> 30% brightness.
    """
    peds = row.get(ped_col, 0)
    vehs = row.get(veh_col, 0)
    
    # Ensure they are numeric
    try:
        peds = float(peds) if not np.isnan(peds) else 0.0
    except:
        peds = 0.0
    try:
        vehs = float(vehs) if not np.isnan(vehs) else 0.0
    except:
        vehs = 0.0
        
    if peds > 30 or vehs > 15:
        return 100.0
    elif peds > 10 or vehs > 5:
        return 60.0
    else:
        return 30.0

def adaptive_ai_control_policy(row, pred_ped, pred_veh, weather_boost_col='Recommended_Brightness_Boost_pct'):
    """
    Adaptive AI Control (AIC) Policy:
    Continuous, optimal dimming controller that maps prediction variables to brightness:
      - Maps predicted pedestrian and vehicle counts using a utility curve:
        B_base(ped, veh) = 30 + 70 * (1 - exp(-0.06 * ped - 0.12 * veh))
      - Weather Boost: Incorporates a dynamic boost factor for safety during poor visibility.
      - Ambient Lux/UV dimming: Reduces brightness if UV_Index/ambient light is high.
    """
    # Safety baseline dimming floor is 30%
    lambda_p = 0.06
    lambda_v = 0.12
    
    # Calculate utility-based continuous brightness
    utility_factor = 1.0 - np.exp(-lambda_p * pred_ped - lambda_v * pred_veh)
    base_brightness = 30.0 + 70.0 * utility_factor
    
    # Apply weather boost factor
    weather_boost = row.get(weather_boost_col, 0.0)
    try:
        weather_boost = float(weather_boost) if not np.isnan(weather_boost) else 0.0
    except:
        weather_boost = 0.0
        
    brightness = base_brightness + weather_boost
    
    # Apply Ambient Light Dimming if UV index is high (e.g. twilight/dawn)
    uv_index = row.get('UV_Index', 0)
    try:
        uv_index = float(uv_index) if not np.isnan(uv_index) else 0.0
    except:
        uv_index = 0.0
        
    if uv_index > 1.0:
        # Scale down brightness during twilight/day transition
        reduction = min(20.0, uv_index * 5.0)
        brightness -= reduction
        
    # Keep brightness bounded between 30% and 100%
    return max(30.0, min(100.0, brightness))
