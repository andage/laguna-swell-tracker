import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="LBsurf", layout="wide")

# Station Catalog
STATIONS = {
    "028 - San Pedro (Outer OC/LA)": "028",
    "271 - Green Beach Offshore (Camp Pendleton/San Onofre)": "271",
    "220 - Mission Bay West (San Diego)": "220",
    "092 - San Pedro South": "092",
    "045 - Oceanside Offshore": "045"
}

# Sidebar Controls
st.sidebar.header("Buoy & Data Configuration")
selected_station_label = st.sidebar.selectbox("Select Station", list(STATIONS.keys()))
station_id = STATIONS[selected_station_label]

duration_hours = st.sidebar.slider("Sample Window (Hours)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

st.sidebar.header("Bandpass Filtering (Swell Band)")
period_min = st.sidebar.number_input("Min Period (sec)", min_value=8.0, max_value=25.0, value=14.0, step=1.0)
period_max = st.sidebar.number_input("Max Period (sec)", min_value=10.0, max_value=30.0, value=22.0, step=1.0)

st.sidebar.header("Directional Window (Brooks / Laguna)")
dir_min = st.sidebar.number_input("Min Approach Direction (° True)", 0, 360, 190)
dir_max = st.sidebar.number_input("Max Approach Direction (° True)", 0, 360, 220)

@st.cache_data(ttl=900)
def fetch_and_process_cdip(station, hours, p_min, p_max, d_min, d_max):
    url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    try:
        ds = xr.open_dataset(url)
    except Exception as e:
        return None, f"Failed to connect to CDIP OpenDAP server: {e}"

    fs = float(ds.xyzSampleRate.values)
    total_samples = int(hours * 3600 * fs)
    
    # Subsample stride = 2 (~0.64 Hz) to cut memory load while capturing up to 4s chop
    stride = 2
    eff_fs = fs / stride
    
    try:
        z_raw = ds.xyzZDisplacement[-total_samples::stride].values.astype(np.float64)
        x_raw = ds.xyzXDisplacement[-total_samples::stride].values.astype(np.float64)
        y_raw = ds.xyzYDisplacement[-total_samples::stride].values.astype(np.float64)
    except Exception as e:
        return None, f"Error pulling array slice: {e}"

    # Handle missing/fill values
    fill_mask = (z_raw < -900) | (x_raw < -900) | (y_raw < -900)
    z_raw[fill_mask] = 0.0
    x_raw[fill_mask] = 0.0
    y_raw[fill_mask] = 0.0

    n_pts = len(z_raw)
    time_min = np.arange(n_pts) / (eff_fs * 60.0)

    # 1. 4th-Order Butterworth Bandpass Filter
    lowcut = 1.0 / p_max
    highcut = 1.0 / p_min
    b, a = butter(4, [lowcut, highcut], btype="band", fs=eff_fs)
    
    z_filt = filtfilt(b, a, z_raw)
    x_filt = filtfilt(b, a, x_raw)
    y_filt = filtfilt(b, a, y_raw)

    # 2. Wave Envelope Extraction (Hilbert Transform)
    z_env = np.abs(hilbert(z_filt))

    # 3. Peak Detection on Envelope (Packets)
    # Require minimum packet spacing of ~60 seconds to avoid double-counting waves within a single set
    min_dist_samples = int(60.0 * eff_fs)
    # Set threshold: average envelope + 0.8 std dev
    threshold = float(np.mean(z_env) + 0.8 * np.std(z_env))
    
    peaks, props = find_peaks(z_env, height=threshold, distance=min_dist_samples)

    # 4. Instantaneous Direction Calculation per Packet
    # CDIP Convention: X = North, Y = West.
    # Buoy heaves towards particle motion; direction of wave arrival is atan2(-Y, -X) converted to compass deg
    packets = []
    for p in peaks:
        # Window around envelope crest (+/- 8 seconds)
        w = int(8.0 * eff_fs)
        idx_s = max(0, p - w)
        idx_e = min(n_pts, p + w)
        
        # Max heave and horizontal correlation in packet
        z_crest = np.max(z_filt[idx_s:idx_e])
        z_trough = np.min(z_filt[idx_s:idx_e])
        packet_height = (z_crest - z_trough) * 3.28084  # meters to feet

        dx = np.mean(x_filt[idx_s:idx_e])
        dy = np.mean(y_filt[idx_s:idx_e])
        
        # Calculate compass propagation angle (degrees true)
        angle_rad = np.arctan2(-dy, -dx)
        angle_deg = (np.degrees(angle_rad) + 360) % 360
        
        # Validate against directional window
        is_valid = False
        if d_min <= d_max:
            is_valid = d_min <= angle_deg <= d_max
        else: # Handle wrapping 360
            is_valid = angle_deg >= d_min or angle_deg <= d_max

        packets.append({
            "index": p,
            "time_min": time_min[p],
            "height_ft": packet_height,
            "direction": angle_deg,
            "valid": is_valid
        })

    return {
        "time_min": time_min,
        "z_filt": z_filt * 3.28084,
        "z_env": z_env * 3.28084,
        "threshold": threshold * 3.28084,
        "packets": packets
    }, None

# Execution & Display
st.title("🏄 Real-Time Swell Set Consistency Analyzer")
st.caption(f"Connected to Scripps CDIP THREDDS OpenDAP | Station {station_id}")

with st.spinner("Fetching 3D displacement vectors and calculating packet consistency..."):
    data, err = fetch_and_process_cdip(station_id, duration_hours, period_min, period_max, dir_min, dir_max)

if err:
    st.error(err)
else:
    all_packets = data["packets"]
    valid_packets = [p for p in all_packets if p["valid"]]

    # Calculate Lulls & Spacing
    if len(valid_packets) > 1:
        intervals = [valid_packets[i+1]["time_min"] - valid_packets[i]["time_min"] for i in range(len(valid_packets)-1)]
        avg_lull = np.mean(intervals)
        min_lull = np.min(intervals)
        max_lull = np.max(intervals)
        avg_set_height = np.mean([p["height_ft"] for p in valid_packets])
    else:
        avg_lull = min_lull = max_lull = avg_set_height = 0.0

    # Key Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Lull Between Sets", f"{avg_lull:.1f} min" if avg_lull > 0 else "N/A")
    col2.metric("Lull Range (Min / Max)", f"{min_lull:.1f} / {max_lull:.1f} min" if avg_lull > 0 else "N/A")
    col3.metric("Deepwater Set Height", f"{avg_set_height:.1f} ft" if avg_set_height > 0 else "N/A")
    col4.metric("Valid Sets Count", f"{len(valid_packets)} sets")

    # Time-Domain Waveform Plot
    fig = go.Figure()

    # Filtered displacement trace
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_filt"], 
        mode="lines", 
        name=f"Filtered Heave ({period_min:.0f}-{period_max:.0f}s)",
        line=dict(color="#29b6f6", width=1.0)
    ))

    # Hilbert envelope
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_env"], 
        mode="lines", 
        name="Wave Envelope",
        line=dict(color="#ff9800", width=1.5, dash="dash")
    ))

    # Set threshold line
    fig.add_hline(
        y=data["threshold"], 
        line=dict(color="#f44336", dash="dot", width=1), 
        annotation_text="Set Trigger Threshold"
    )

    # Valid sets marker (Target Direction)
    if valid_packets:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in valid_packets],
            y=[p["height_ft"]/2.0 for p in valid_packets],
            mode="markers+text",
            name="Target Groundswell Set",
            text=[f"{p['height_ft']:.1f}ft @ {p['direction']:.0f}°" for p in valid_packets],
            textposition="top center",
            marker=dict(color="#00e676", size=10, symbol="diamond")
        ))

    # Rejected packets marker (Chop/Shadowed)
    rejected_packets = [p for p in all_packets if not p["valid"]]
    if rejected_packets:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in rejected_packets],
            y=[p["height_ft"]/2.0 for p in rejected_packets],
            mode="markers",
            name="Rejected Set (Out of Angle)",
            marker=dict(color="#757575", size=6, symbol="x")
        ))

    fig.update_layout(
        title="Time-Series Displacement & Wave Group Envelopes",
        xaxis_title="Elapsed Time (Minutes)",
        yaxis_title="Displacement (Feet)",
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Set Log Breakdown
    if valid_packets:
        st.subheader("Detected Groundswell Sets")
        packet_rows = []
        for i, p in enumerate(valid_packets):
            lull = f"{(p['time_min'] - valid_packets[i-1]['time_min']):.1f} min" if i > 0 else "—"
            packet_rows.append({
                "Set #": i + 1,
                "Arrival Time": f"+{p['time_min']:.1f} min",
                "Wait / Lull Since Prior": lull,
                "Estimated Offshore Set Height": f"{p['height_ft']:.2f} ft",
                "Approach Bearing": f"{p['direction']:.1f}° True"
            })
        st.table(packet_rows)
