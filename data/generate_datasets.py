import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_raw_header_excel(file_path, df, title, description):
    """
    Creates an Excel file with the standard multi-row header structure:
    Row 0: Title
    Row 1: Description
    Row 2: Column Names
    Row 3+: Data
    """
    # Create a wrapper DataFrame to write
    header_rows = [
        [title] + [np.nan] * (len(df.columns) - 1),
        [description] + [np.nan] * (len(df.columns) - 1),
        list(df.columns)
    ]
    
    # Concatenate header lists and data
    data_list = df.values.tolist()
    all_rows = header_rows + data_list
    
    new_df = pd.DataFrame(all_rows)
    new_df.to_excel(file_path, index=False, header=False)
    print(f"Saved {file_path} with {len(df)} rows.")

def generate_all_datasets():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # ------------------ 1. Nodes (120 Nodes) ------------------
    print("Generating 120 Nodes...")
    num_nodes = 120
    node_ids = [f"Node_{i:03d}" for i in range(1, num_nodes + 1)]
    
    # Coordinate center around Casablanca/Settat area (32.9, -7.6)
    latitudes = np.random.uniform(32.90, 33.00, num_nodes)
    longitudes = np.random.uniform(-7.65, -7.50, num_nodes)
    
    node_types = np.random.choice(["transformer", "electrical_cabinet", "pole"], size=num_nodes, p=[0.1, 0.15, 0.75])
    sectors = [f"Sector_{chr(65 + (i % 7))}" for i in range(num_nodes)] # Sector_A to Sector_G
    streets = [
        "Lalla Aicha", "Avenue OCP", "Ibn Sina", "Rue Al Amal", "Al Fida", 
        "Rue Hassan II", "Boulevard Mohammed V", "Avenue des FAR", "Rue de Fes", "Avenue Tarik"
    ]
    street_assignments = [streets[i % len(streets)] for i in range(num_nodes)]
    zones = np.random.choice(["residential", "commercial", "industrial", "park"], size=num_nodes, p=[0.4, 0.3, 0.2, 0.1])
    
    nodes_df = pd.DataFrame({
        "Node_ID": node_ids,
        "Latitude": latitudes,
        "Longitude": longitudes,
        "Node_Type": node_types,
        "Sector": sectors,
        "Street": street_assignments,
        "Zone": zones,
        "Installation_Date": [datetime(2022, 1, 1) + timedelta(days=int(np.random.randint(0, 1000))) for _ in range(num_nodes)],
        "Last_Maintenance": [datetime(2025, 6, 1) + timedelta(days=int(np.random.randint(0, 300))) for _ in range(num_nodes)],
        "Elevation_m": np.random.uniform(550.0, 650.0, num_nodes).round(1),
        "Height_m": np.random.uniform(7.5, 12.0, num_nodes).round(1),
        "Connected_Grid": ["Yes"] * num_nodes,
        "Has_Sensor": np.random.choice(["Yes", "No"], size=num_nodes, p=[0.8, 0.2]),
        "Has_Camera": np.random.choice(["Yes", "No"], size=num_nodes, p=[0.3, 0.7]),
        "Condition": np.random.choice(["good", "fair", "poor"], size=num_nodes, p=[0.7, 0.25, 0.05]),
        "Contractor": ["Lydec"] * num_nodes
    })
    
    # Format dates
    nodes_df["Installation_Date"] = nodes_df["Installation_Date"].dt.strftime("%Y-%m-%d")
    nodes_df["Last_Maintenance"] = nodes_df["Last_Maintenance"].dt.strftime("%Y-%m-%d")
    
    create_raw_header_excel(
        os.path.join(data_dir, "15_Extended_Nodes.xlsx"),
        nodes_df,
        "15 - Extended Node Registry (120 nodes)",
        "Full topology including zone, street, condition, contractor and geographic coordinates."
    )
    
    # ------------------ 2. Adjacency Matrix (Sparse Graph) ------------------
    print("Generating Adjacency Matrix...")
    edges = []
    # Add a spanning cycle to guarantee connectedness for road and electrical networks
    for i in range(num_nodes):
        u = node_ids[i]
        v = node_ids[(i + 1) % num_nodes]
        edges.append((u, v, "road"))
        edges.append((u, v, "electrical"))
        
    # Add extra random edges to make it a realistic grid (around 400 edges total)
    extra_edges_count = 160
    added = set()
    while len(added) < extra_edges_count:
        i, j = np.random.choice(num_nodes, 2, replace=False)
        u, v = node_ids[min(i, j)], node_ids[max(i, j)]
        edge_type = "road" if np.random.rand() > 0.5 else "electrical"
        key = (u, v, edge_type)
        if key not in added:
            added.add(key)
            edges.append((u, v, edge_type))
            
    adj_records = []
    for u, v, etype in edges:
        # Distance calculation in meters
        lat1, lon1 = nodes_df.loc[nodes_df["Node_ID"] == u, ["Latitude", "Longitude"]].values[0]
        lat2, lon2 = nodes_df.loc[nodes_df["Node_ID"] == v, ["Latitude", "Longitude"]].values[0]
        dist_m = np.sqrt(((lat1 - lat2) * 111000)**2 + ((lon1 - lon2) * 93000)**2).round(1)
        
        # Weight is inversely proportional to distance
        weight = (1.0 / (dist_m + 10.0)).round(5)
        bidirectional = "Yes" if etype == "road" else "No"
        
        adj_records.append({
            "Node_i": u,
            "Node_j": v,
            "Weight": weight,
            "Distance_m": dist_m,
            "Edge_Type": etype,
            "Bidirectional": bidirectional
        })
        
    adj_df = pd.DataFrame(adj_records)
    create_raw_header_excel(
        os.path.join(data_dir, "16_Adjacency_Matrix.xlsx"),
        adj_df,
        "16 - Graph Adjacency Matrix (Sparse)",
        "Edge weights for STGCN graph convolution - weights are distance-inversed."
    )
    
    # ------------------ 3. Historical Data Setup (6 Months) ------------------
    print("Generating 6 Months of hourly data...")
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 6, 30, 23, 0)
    
    # Generate Timestamp hourly range
    timestamps = []
    curr = start_date
    while curr <= end_date:
        timestamps.append(curr)
        curr += timedelta(hours=1)
        
    num_hours = len(timestamps)
    print(f"Total hours: {num_hours}")
    
    # ------------------ 4. Weather Logs ------------------
    print("Generating Weather Logs...")
    # Seasonal temperature: starts cold in Jan, peaks in June
    temp_profile = 10.0 + 15.0 * np.sin(2 * np.pi * np.arange(num_hours) / (365 * 24))
    # Daily fluctuation
    daily_temp = 5.0 * np.sin(2 * np.pi * np.array([ts.hour for ts in timestamps]) / 24.0)
    temp_c = (temp_profile + daily_temp + np.random.normal(0, 1.5, num_hours)).round(1)
    
    humidity = np.clip(80.0 - 1.2 * temp_c + np.random.normal(0, 5.0, num_hours), 20, 100).round(1)
    wind_speed = np.clip(10.0 + np.random.normal(0, 4.0, num_hours), 0, 45).round(1)
    pressure = (1013.25 - 0.1 * temp_c + np.random.normal(0, 2.0, num_hours)).round(1)
    
    # Rain generation: based on humidity
    rain_prob = (humidity / 100.0) ** 4
    rain_mm = np.where(np.random.rand(num_hours) < rain_prob * 0.15, np.random.uniform(0.1, 5.0, num_hours), 0.0).round(1)
    
    # UV index: daily peaks, high in summer
    hour_vals = np.array([ts.hour for ts in timestamps])
    sun_intensity = np.clip(np.sin(2 * np.pi * (hour_vals - 6) / 24.0), 0, 1)
    uv_index = (10.0 * sun_intensity * (1.0 + np.sin(2 * np.pi * np.arange(num_hours) / (365 * 24))) / 2.0).round(1)
    uv_index = np.where(sun_intensity == 0, 0.0, uv_index)
    
    visibility = np.clip(10.0 - rain_mm * 1.5 - (humidity - 80) * 0.1 + np.random.normal(0, 0.5, num_hours), 1.0, 10.0).round(1)
    
    # Brightness Boost: 15% if raining, 10% if fog/low visibility
    boost = np.zeros(num_hours)
    boost = np.where(rain_mm > 0.5, 15.0, boost)
    boost = np.where((visibility < 5.0) & (rain_mm <= 0.5), 10.0, boost)
    boost = np.where(visibility < 2.0, 25.0, boost)
    
    weather_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Temp_C": temp_c,
        "Feels_Like_C": (temp_c + 0.1 * (humidity - 50.0)).round(1),
        "Humidity_pct": humidity,
        "Wind_Speed_kmh": wind_speed,
        "Pressure_hPa": pressure,
        "Rain_mm": rain_mm,
        "UV_Index": uv_index,
        "Visibility_km": visibility,
        "Recommended_Brightness_Boost_pct": boost.round(1)
    })
    
    create_raw_header_excel(
        os.path.join(data_dir, "22_Weather_30days.xlsx"), # Keep filename as referenced in loader
        weather_df,
        "22 - Weather Logs (Hourly, 6 Months)",
        "Weather parameters and recommended safety brightness boost levels."
    )
    
    # ------------------ 5. Pedestrian & Vehicle Traffic ------------------
    # To avoid Excel file size overflow and make STGCN training fast, let's generate 
    # hourly traffic for the 120 nodes. For 180 days (4344 hours) and 120 nodes, that is 521,280 rows.
    # We will save this as a CSV/Excel file. Writing to Excel takes about 20-30 seconds.
    print("Generating Traffic Logs...")
    traffic_records = []
    
    # Pre-generate daily profiles
    # Rush hours: 7-9 and 17-19
    # Night: 23-5
    # Weekday vs Weekend
    
    # We will generate node-by-node to make it fast
    # Node features
    node_zones = nodes_df["Zone"].values
    node_sectors = nodes_df["Sector"].values
    
    timestamps_series = pd.Series(timestamps)
    hour_arr = timestamps_series.dt.hour.values
    dow_arr = timestamps_series.dt.dayofweek.values
    is_weekend = np.isin(dow_arr, [5, 6]).astype(int)
    
    # Ramadan effect: approx 30 days (let's say March 10 to April 9)
    is_ramadan = ((timestamps_series.dt.month == 3) & (timestamps_series.dt.day >= 10)) | ((timestamps_series.dt.month == 4) & (timestamps_series.dt.day <= 9))
    is_ramadan = is_ramadan.astype(int).values
    
    # Daily profile multipliers
    # Night, Morning Rush, Midday, Evening Rush, Evening
    base_peds = np.zeros(num_hours)
    base_vehs = np.zeros(num_hours)
    
    for h in range(24):
        if h >= 0 and h <= 5:
            base_peds[hour_arr == h] = 2.0
            base_vehs[hour_arr == h] = 1.0
        elif h >= 7 and h <= 9:
            base_peds[hour_arr == h] = 25.0
            base_vehs[hour_arr == h] = 18.0
        elif h >= 17 and h <= 19:
            base_peds[hour_arr == h] = 30.0
            base_vehs[hour_arr == h] = 22.0
        elif h >= 12 and h <= 14:
            base_peds[hour_arr == h] = 12.0
            base_vehs[hour_arr == h] = 10.0
        else:
            base_peds[hour_arr == h] = 8.0
            base_vehs[hour_arr == h] = 6.0
            
    # Weekday/weekend adjustments
    # Weekend: morning rush is lower, evening/night is higher
    weekend_peds = base_peds.copy()
    weekend_vehs = base_vehs.copy()
    
    for h in range(24):
        if h >= 7 and h <= 9:
            weekend_peds[hour_arr == h] *= 0.3
            weekend_vehs[hour_arr == h] *= 0.4
        elif h >= 20 or h <= 2:
            weekend_peds[hour_arr == h] *= 1.8
            weekend_vehs[hour_arr == h] *= 1.5
            
    # Assemble traffic data node-by-node
    dfs = []
    
    for i, node in enumerate(node_ids):
        zone = node_zones[i]
        
        # Sector multiplier
        sect_mult = 1.0 + 0.1 * (i % 5)
        
        # Zone modifiers
        if zone == "industrial":
            zone_peds = base_peds * 0.5
            zone_vehs = base_vehs * 1.5
        elif zone == "commercial":
            zone_peds = base_peds * 1.4
            zone_vehs = base_vehs * 1.2
        elif zone == "park":
            zone_peds = base_peds * 1.6
            zone_vehs = base_vehs * 0.4
        else: # residential
            zone_peds = base_peds
            zone_vehs = base_vehs
            
        # Combine
        peds = np.where(is_weekend == 1, zone_peds * 0.9, zone_peds) * sect_mult
        vehs = np.where(is_weekend == 1, zone_vehs * 0.8, zone_vehs) * sect_mult
        
        # Ramadan effect: late night traffic increases, daytime decreases
        peds = np.where(is_ramadan == 1, np.where((hour_arr >= 20) | (hour_arr <= 2), peds * 2.2, peds * 0.5), peds)
        vehs = np.where(is_ramadan == 1, np.where((hour_arr >= 20) | (hour_arr <= 2), vehs * 1.8, vehs * 0.6), vehs)
        
        # Add noise
        p_noise = np.random.poisson(lam=peds).astype(float)
        v_noise = np.random.poisson(lam=vehs).astype(float)
        
        # Rush flags
        is_rush = ((hour_arr >= 7) & (hour_arr <= 9)) | ((hour_arr >= 17) & (hour_arr <= 19))
        is_rush = is_rush.astype(int)
        
        # Night flags (20:00 to 05:00)
        is_night = (hour_arr >= 20) | (hour_arr <= 5)
        is_night = is_night.astype(int)
        
        node_df = pd.DataFrame({
            "Timestamp": timestamps,
            "Node_ID": [node] * num_hours,
            "Pedestrians": p_noise.astype(int),
            "Vehicles": v_noise.astype(int),
            "Cyclists": (p_noise * 0.1).astype(int),
            "Hour": hour_arr,
            "Minute": [0] * num_hours,
            "Day_of_Week": [["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dow] for dow in dow_arr],
            "Is_Weekend": is_weekend,
            "Is_Rush": is_rush,
            "Is_Night": is_night,
            "Ramadan_Effect": is_ramadan
        })
        dfs.append(node_df)
        
    traffic_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    # Save traffic
    # Excel has a limit of 1,048,576 rows. Our rows = 521,280.
    # To save memory and write fast, we write to Excel directly.
    traffic_file = os.path.join(data_dir, "17_Traffic_30min_30days.xlsx")
    create_raw_header_excel(
        traffic_file,
        traffic_df,
        "17 - Spatio-Temporal Sensor Traffic Streams (6 Months)",
        "Pedestrian, vehicle, and cyclist flows tracked hourly across all 120 grid nodes."
    )
    
    # ------------------ 6. Fault History (500 events) ------------------
    print("Generating Fault History...")
    fault_ids = [f"F_{i:04d}" for i in range(1, 501)]
    lamp_ids = [f"L_{np.random.randint(1, 81):02d}" for _ in range(500)]
    fault_node_ids = np.random.choice(node_ids, size=500)
    
    fault_types = ["short_circuit", "vandalism", "blown_fuse", "corrosion", "voltage_spike", 
                   "controller_failure", "water_infiltration", "lamp_failure", "overheating", "loose_connection"]
    severity_levels = ["low", "medium", "high", "critical"]
    resolutions = ["resolved", "in_progress", "resolved"]
    
    fault_records = []
    for i in range(500):
        node = fault_node_ids[i]
        node_row = nodes_df.loc[nodes_df["Node_ID"] == node].iloc[0]
        fault_ts = datetime(2026, 1, 1) + timedelta(minutes=int(np.random.randint(0, 180 * 24 * 60)))
        
        fault_records.append({
            "Fault_ID": fault_ids[i],
            "Lamp_ID": lamp_ids[i],
            "Node_ID": node,
            "Sector": node_row["Sector"],
            "Street": node_row["Street"],
            "Fault_Type": np.random.choice(fault_types),
            "Severity": np.random.choice(severity_levels, p=[0.4, 0.35, 0.2, 0.05]),
            "Timestamp": fault_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Detection_Method": np.random.choice(["SCADA_alert", "camera", "manual_report", "sensor_telemetry"]),
            "Response_Time_min": int(np.random.randint(15, 180)),
            "Repair_Time_h": round(float(np.random.uniform(0.5, 6.0)), 1),
            "Downtime_h": round(float(np.random.uniform(1.0, 12.0)), 1),
            "Technician_ID": f"TECH_{np.random.randint(1, 6):02d}",
            "Parts_Replaced": np.random.choice(["lamp_unit", "cables", "fuse", "capacitor", "controller", "full_pole"]),
            "Cost_MAD": round(float(np.random.uniform(500.0, 8000.0)), 2),
            "Recurrence": np.random.choice(["Yes", "No"], p=[0.15, 0.85]),
            "Weather_At_Fault": np.random.choice(["clear", "wind", "rain", "fog", "drizzle"]),
            "Resolution": np.random.choice(resolutions),
            "Root_Cause": np.random.choice(["vandalism", "aging_infrastructure", "weather_damage", "grid_instability"])
        })
        
    fault_df = pd.DataFrame(fault_records)
    create_raw_header_excel(
        os.path.join(data_dir, "20_Fault_History_500.xlsx"),
        fault_df,
        "20 - Fault History (500 events)",
        "Extended maintenance log with root cause, cost, severity and resolution timestamps."
    )
    
    # ------------------ 7. NLP Fault Reports (300 reports) ------------------
    print("Generating NLP Reports...")
    nlp_ids = [f"RPT_{i:04d}" for i in range(1, 301)]
    nlp_langs = ["French", "Arabic", "Darija"]
    
    # Multilingual templates
    french_templates = [
        "Armoire electrique CBL_{num} ouverte vandalisee, cables exposes.",
        "Luminaire L_{num} clignote, temperature anormale 65C relevee.",
        "Alimentation instable secteur Sector_{sec}, coupures frequentes depuis 3 jours.",
        "Explosion fusible principal, poteau Node_{num} totalement eteint.",
        "Surtension reseau secteur Sector_{sec}, transformateur T_{sec_num} surcharge.",
        "Poteau Node_{num} penche suite a un accident de voiture, danger public.",
        "Le lampadaire L_{num} reste allume en plein jour, anomalie de capteur.",
        "Court-circuit detecte poteau Node_{num}, secteur Sector_{sec}, intervention urgente.",
        "Corrosion severe detectee a la base du poteau Node_{num}, risque d'effondrement.",
        "Infiltration d'eau dans le boitier electrique du poteau Node_{num} apres la pluie."
    ]
    
    arabic_templates = [
        "انقطاع الكهرباء في منطقة Sector_{sec}، المحول T_{sec_num} متوقف عن العمل.",
        "العمود الكهربائي Node_{num} مائل بشكل خطير، يرجى التدخل.",
        "المصباح L_{num} لا يعمل، الشارع مظلم تماما.",
        "تخريب في الخزانة الكهربائية التابعة للموقع Node_{num}، الأسلاك عارية.",
        "حرارة مفرطة في المحول Node_{num} مع سماع صوت غير طبيعي.",
        "تسرب المياه لداخل عمود الإنارة Node_{num} بسبب المطر."
    ]
    
    darija_templates = [
        "Node_{num} daro chi had vandal, l-armoire mftoha.",
        "3amoud Node_{num} f Sector_{sec} khayb, l'eclairage machi kaydwi.",
        "Cable d Al Fida mherres, khesshom yji ysalho bsraa.",
        "L-poteau Node_{num} tayh binsba l-hdida, aji chofoh.",
        "Lampadaire L_{num} makaytfech bnhar, khasser capteur.",
        "Short-circuit f l-boite dyal poteau Node_{num}, ch3lat fih l-3afya."
    ]
    
    nlp_records = []
    for i in range(300):
        lang = np.random.choice(nlp_langs, p=[0.5, 0.25, 0.25])
        sec_char = chr(65 + np.random.randint(0, 7))
        num_val = f"{np.random.randint(1, 121):03d}"
        lamp_num = f"{np.random.randint(1, 81):03d}"
        sec_num = f"{np.random.randint(1, 5):02d}"
        
        if lang == "French":
            raw_text = np.random.choice(french_templates).format(num=num_val, sec=sec_char, sec_num=sec_num)
        elif lang == "Arabic":
            raw_text = np.random.choice(arabic_templates).format(num=num_val, sec=sec_char, sec_num=sec_num)
        else:
            raw_text = np.random.choice(darija_templates).format(num=num_val, sec=sec_char, sec_num=sec_num)
            
        nlp_records.append({
            "Report_ID": nlp_ids[i],
            "Timestamp": (datetime(2026, 4, 1) + timedelta(minutes=int(np.random.randint(0, 90 * 24 * 60)))).strftime("%Y-%m-%d %H:%M:%S"),
            "Channel": np.random.choice(["mobile_app", "call_center", "web_portal"]),
            "Language": lang,
            "Raw_Report": raw_text,
            "Extracted_Node": f"Node_{np.random.randint(1, 121):03d}",
            "Extracted_Lamp": f"L_{np.random.randint(1, 81):03d}",
            "Extracted_Fault": np.random.choice(fault_types),
            "Confidence_Score": round(float(np.random.uniform(0.5, 0.99)), 3),
            "Urgency_Score": round(float(np.random.uniform(0.1, 0.99)), 3),
            "Sentiment": np.random.choice(severity_levels),
            "Auto_Dispatched": np.random.choice(["Yes", "No"], p=[0.7, 0.3]),
            "Linked_Fault_ID": f"F_{np.random.randint(1, 501):04d}",
            "NLP_Model_Version": "bert-multilingual-v2.1"
        })
        
    nlp_df = pd.DataFrame(nlp_records)
    create_raw_header_excel(
        os.path.join(data_dir, "21_NLP_Reports_300.xlsx"),
        nlp_df,
        "21 - NLP Fault Reports (300 - FR/AR/Darija)",
        "Multilingual unstructured reports with NLP extracted entity metadata."
    )
    
    # ------------------ 8. Predictive Maintenance (80 Lamps) ------------------
    print("Generating Predictive Maintenance Scores...")
    lamp_ids_list = [f"L_{i:03d}" for i in range(1, 81)]
    lamp_nodes = np.random.choice(node_ids, size=80, replace=False)
    
    pm_records = []
    for i in range(80):
        lamp_id = lamp_ids_list[i]
        node = lamp_nodes[i]
        node_sec = nodes_df.loc[nodes_df["Node_ID"] == node, "Sector"].values[0]
        
        pm_records.append({
            "Lamp_ID": lamp_id,
            "Node_ID": node,
            "Sector": node_sec,
            "Age_Days": int(np.random.randint(100, 4000)),
            "Cumulative_kWh": round(float(np.random.uniform(100.0, 3000.0)), 1),
            "Fault_Count_Lifetime": int(np.random.randint(0, 20)),
            "Last_Fault_Days_Ago": int(np.random.randint(5, 300)),
            "Avg_Temp_C": round(float(np.random.uniform(35.0, 58.0)), 1),
            "Voltage_Stability_Score": round(float(np.random.uniform(50.0, 99.0)), 1),
            "Failure_Probability_30d": round(float(np.random.uniform(0.01, 0.85)), 4),
            "RUL_Days_Estimate": int(np.random.randint(10, 1000)),
            "Maintenance_Priority": np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
            "Last_Maintenance": (datetime(2025, 1, 1) + timedelta(days=int(np.random.randint(0, 300)))).strftime("%Y-%m-%d"),
            "Recommended_Action": np.random.choice(["monitor", "schedule_maintenance", "replace_lamp", "replace_controller"]),
            "Anomaly_Score": round(float(np.random.uniform(0.05, 0.85)), 4),
            "Isolation_Forest_Label": np.random.choice(["normal", "anomaly"], p=[0.9, 0.1])
        })
        
    pm_df = pd.DataFrame(pm_records)
    create_raw_header_excel(
        os.path.join(data_dir, "25_Predictive_Maintenance.xlsx"),
        pm_df,
        "25 - Predictive Maintenance Scores",
        "RUL estimation, failure probability, anomaly score and recommended maintenance actions."
    )
    
    # ------------------ 9. Route Optimization (200 Routes) ------------------
    print("Generating Route Optimization data...")
    route_ids = [f"RTE_{i:03d}" for i in range(1, 201)]
    route_records = []
    
    for i in range(200):
        src, dest = np.random.choice(node_ids, 2, replace=False)
        hops = int(np.random.randint(1, 8))
        path_nodes = "?".join(list(np.random.choice(node_ids, size=hops))) + "?"
        
        route_records.append({
            "Route_ID": route_ids[i],
            "Source_Node": src,
            "Destination_Node": dest,
            "Algorithm": np.random.choice(["Dijkstra", "Bellman-Ford"], p=[0.6, 0.4]),
            "Path_Nodes": path_nodes,
            "Hops": hops,
            "Total_Distance_m": round(float(np.random.uniform(100.0, 1500.0)), 1),
            "Estimated_Time_min": round(float(np.random.uniform(1.0, 15.0)), 1),
            "Purpose": "maintenance_dispatch",
            "Priority": np.random.choice(["low", "medium", "high"]),
            "Computed_At": (datetime(2026, 4, 1) + timedelta(days=int(np.random.randint(0, 30)))).strftime("%Y-%m-%d")
        })
        
    route_df = pd.DataFrame(route_records)
    create_raw_header_excel(
        os.path.join(data_dir, "26_Route_Optimization.xlsx"),
        route_df,
        "26 - Route Optimization (Bellman-Ford / Dijkstra)",
        "Optimal maintenance dispatch paths across the graph computed dynamically."
    )
    
    # ------------------ 10. Daily KPI Summary (180 days) ------------------
    print("Generating Daily KPI Summary...")
    days_range = pd.date_range(start="2026-01-01", end="2026-06-30")
    kpi_records = []
    
    for idx, dt in enumerate(days_range):
        date_str = dt.strftime("%Y-%m-%d")
        day_str = dt.strftime("%A")
        is_wk = 1 if day_str in ["Saturday", "Sunday"] else 0
        
        # Initial values (will be filled after running optimizations)
        kpi_records.append({
            "Date": date_str,
            "Day": day_str,
            "Is_Weekend": is_wk,
            "Total_Energy_kWh": 0.0,
            "Energy_Saved_vs_Baseline_kWh": 0.0,
            "Savings_pct": 0.0,
            "Avg_Brightness_pct": 100.0,
            "Active_Lamps": 80,
            "Faulty_Lamps": int(np.random.randint(0, 5)),
            "Total_Faults": int(np.random.randint(0, 10)),
            "Critical_Faults": int(np.random.choice([0, 1], p=[0.8, 0.2])),
            "Avg_Response_Time_min": round(float(np.random.uniform(30, 120)), 1),
            "Peak_Demand_kW": round(float(np.random.uniform(8.0, 12.0)), 1),
            "Avg_Load_Factor_pct": round(float(np.random.uniform(60, 85)), 1),
            "Total_Pedestrians": int(np.random.randint(8000, 20000)),
            "Total_Vehicles": int(np.random.randint(5000, 15000)),
            "CO2_Saved_kg": 0.0,
            "Grid_Stability_Score": round(float(np.random.uniform(95.0, 99.9)), 2),
            "STGCN_MAE": 0.0,
            "STGCN_RMSE": 0.0,
            "Revenue_Loss_MAD": round(float(np.random.uniform(100.0, 1000.0)), 2)
        })
        
    kpi_df = pd.DataFrame(kpi_records)
    create_raw_header_excel(
        os.path.join(data_dir, "23_Daily_KPI_Summary.xlsx"),
        kpi_df,
        "23 - Daily KPI Summary Dashboard",
        "30-day aggregated metrics - energy, faults, traffic, stability."
    )
    
    print("ALL DATASETS GENERATED SUCCESSFULLY IN data/ FOLDER!")

if __name__ == "__main__":
    generate_all_datasets()
