import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go

st.set_page_config(page_title="LB Surf", page_icon="🏄", layout="wide")

# Updated Buoy Stations
STATIONS = {
    "092 - San Pedro South": "092",
    "271 - Green Beach Offshore (Camp Pendleton/San Clemente)": "271",
    "220 - Mission Bay West (San Diego)": "220",
    "045 - Oceanside Offshore": "045"
}

# Top Navigation / Controls Bar
col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_station_label = st.selectbox("Select Station", list(STATIONS.keys()), index=0)
    station_id = STATIONS[selected_station_label]
with col_sel2:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Real-Time Data"):
        st.cache_data.clear()
        st.rerun()

# Swell Filtering Configuration
with st.sidebar:
    st.header("Swell Band & Approach Window")
    period_min = st.number_input("Min Period (s)", min_value=8.0, max_value=25.0, value=14.0, step=1.0)
    period_max = st.number_input("Max Period (s)", min_value=10.0, max_value=30.0, value=22.0, step=1.0)
    dir_min = st.number_input("Brooks Window Min (° True)", 0, 360, 190)
    dir_max = st.number_input("Brooks Window Max (° True)", 0, 360, 220)

@st.cache_data(ttl=900)
def fetch_and_process_cdip(station, p_min, p_max, d_min, d_max):
    # Fixed 4-hour window
    hours = 4.0
    url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    try:
        ds = xr.open_dataset(url)
    except Exception as e:
        return None, f"Failed to connect to CDIP OpenDAP endpoint: {e}"

    fs = float(ds.xyzSampleRate.values)
    total_samples = int(hours * 3600 * fs)
    
    # Subsample stride 2 (~0.64 Hz)
    stride = 2
    eff_fs = fs / stride
    
    try:
        z_raw = ds.xyzZDisplacement[-total_samples::stride].values.astype(np.float64)
        x_raw = ds.xyzXDisplacement[-total_samples::stride].values.astype(np.float64)
        y_raw = ds.xyzYDisplacement[-total_samples::stride].values.astype(np.float64)
    except Exception as e:
        return None, f"Error reading displacement arrays: {e}"

    # Clean fill values
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

    # Optional visual smoothing on envelope
    smooth_win = int(eff_fs * 8.0)
    if smooth_win > 1:
        kernel = np.hanning(smooth_win)
        kernel /= np.sum(kernel)
        z_env_smooth = np.convolve(z_env, kernel, mode='same')
    else:
        z_env_smooth = z_env

    # 3. Peak Detection on Envelope
    min_dist_samples = int(70.0 * eff_fs)
    threshold = float(np.mean(z_env) + 0.75 * np.std(z_env))
    peaks, _ = find_peaks(z_env, height=threshold, distance=min_dist_samples)

    # 4. Instantaneous Direction & Wave Count per Packet
    packets = []
    for p in peaks:
        # Search window for the packet waves (+/- 25 seconds)
        w = int(25.0 * eff_fs)
        idx_s = max(0, p - w)
        idx_e = min(n_pts, p + w)
        
        # Estimate wave count within the packet
        sub_peaks, _ = find_peaks(z_filt[idx_s:idx_e], distance=int(eff_fs * p_min * 0.7))
        wave_count = max(len(sub_peaks), 1)

        z_crest = np.max(z_filt[idx_s:idx_e])
        z_trough = np.min(z_filt[idx_s:idx_e])
        packet_height = (z_crest - z_trough) * 3.28084  # Convert to feet

        dx = np.mean(x_filt[idx_s:idx_e])
        dy = np.mean(y_filt[idx_s:idx_e])
        
        angle_rad = np.arctan2(-dy, -dx)
        angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0
        
        if d_min <= d_max:
            is_valid = d_min <= angle_deg <= d_max
        else:
            is_valid = angle_deg >= d_min or angle_deg <= d_max

        packets.append({
            "index": p,
            "time_min": time_min[p],
            "height_ft": packet_height,
            "direction": angle_deg,
            "waves": wave_count,
            "valid": is_valid
        })

    return {
        "time_min": time_min,
        "z_filt": z_filt * 3.28084,
        "z_env_smooth": z_env_smooth * 3.28084,
        "threshold": threshold * 3.28084,
        "packets": packets
    }, None

# UI Header
st.title("🏄 LB Surf Consistency Dashboard")
st.caption(f"Real-Time CDIP Directional Decomposition | Analyzing Station {station_id}")

with st.spinner("Processing 4-hour 3D wave displacement..."):
    data, err = fetch_and_process_cdip(station_id, period_min, period_max, dir_min, dir_max)

if err:
    st.error(err)
else:
    all_packets = data["packets"]
    valid_packets = [p for p in all_packets if p["valid"]]

    if len(valid_packets) > 1:
        intervals = [valid_packets[i+1]["time_min"] - valid_packets[i]["time_min"] for i in range(len(valid_packets)-1)]
        avg_lull = np.mean(intervals)
        min_lull = np.min(intervals)
        max_lull = np.max(intervals)
        avg_set_height = np.mean([p["height_ft"] for p in valid_packets])
        avg_dir = np.mean([p["direction"] for p in valid_packets])
        avg_waves = int(np.round(np.mean([p["waves"] for p in valid_packets])))
    elif len(valid_packets) == 1:
        avg_lull = min_lull = max_lull = 0.0
        avg_set_height = valid_packets[0]["height_ft"]
        avg_dir = valid_packets[0]["direction"]
        avg_waves = valid_packets[0]["waves"]
    else:
        avg_lull = min_lull = max_lull = avg_set_height = avg_dir = avg_waves = 0.0

    # Top Metric Tiles
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Lull Between Sets", f"{avg_lull:.1f} min" if avg_lull > 0 else "N/A")
    m2.metric("Lull Range (Min / Max)", f"{min_lull:.1f} / {max_lull:.1f} min" if avg_lull > 0 else "N/A")
    m3.metric("Deepwater Set Height", f"{avg_set_height:.1f} ft @ {avg_dir:.0f}°" if avg_set_height > 0 else "N/A")
    m4.metric("Sets Detected", f"{len(valid_packets)} sets", delta="Past 4.0 Hours", delta_color="normal")

    # Waveform Plot with Bottom Legend
    fig = go.Figure()

    # Groundswell Heave (Semi-transparent to declutter)
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_filt"], 
        mode="lines", 
        name=f"Filtered Heave ({period_min:.0f}-{period_max:.0f}s)",
        line=dict(color="rgba(41, 182, 246, 0.45)", width=1.0),
        hoverinfo="skip"
    ))

    # Smoothed Envelope
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_env_smooth"], 
        mode="lines", 
        name="Smoothed Wave Envelope",
        line=dict(color="#ff9800", width=2.0)
    ))

    # Trigger Line
    fig.add_hline(
        y=data["threshold"], 
        line=dict(color="#ef5350", dash="dot", width=1.2), 
        annotation_text="Set Threshold",
        annotation_position="bottom left"
    )

    # Valid Groundswell Sets
    if valid_packets:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in valid_packets],
            y=[p["height_ft"]/2.0 for p in valid_packets],
            mode="markers",
            name="Target Groundswell Set",
            marker=dict(color="#00e676", size=10, symbol="diamond", line=dict(width=1, color="#ffffff")),
            hovertemplate="<b>Set Packet</b><br>Time: +%{x:.1f} min<br>Set Height: %{customdata[0]:.2f} ft<br>Waves: ~%{customdata[1]} waves<br>Bearing: %{customdata[2]:.1f}° True<extra></extra>",
            customdata=[[p["height_ft"], p["waves"], p["direction"]] for p in valid_packets]
        ))

    # Decluttered Layout: Bottom Legend & Optimized Margins
    fig.update_layout(
        xaxis_title="Elapsed Time (Minutes)",
        yaxis_title="Surface Displacement (Feet)",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Set Log
    if valid_packets:
        st.subheader("Groundswell Set Log (Target Window)")
        rows = []
        for i, p in enumerate(valid_packets):
            wait = f"{(p['time_min'] - valid_packets[i-1]['time_min']):.1f} min" if i > 0 else "—"
            rows.append({
                "Set #": i + 1,
                "Arrival": f"+{p['time_min']:.1f} min",
                "Lull Duration": wait,
                "Deepwater Height": f"{p['height_ft']:.2f} ft",
                "Waves in Set": f"~{p['waves']} waves",
                "Approach Bearing": f"{p['direction']:.1f}° True"
            })
        st.table(rows)
    else:
        st.info("No sets crossed the threshold within your directional window over the last 4 hours.")
