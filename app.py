import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go

st.set_page_config(page_title="LB Surf", page_icon="🏄", layout="wide")

# Updated Buoy Catalog
STATIONS = {
    "092 - San Pedro South": "092",
    "271 - Green Beach Offshore (Camp Pendleton/San Clemente)": "271",
    "220 - Mission Bay West (San Diego)": "220",
    "045 - Oceanside Offshore": "045"
}

# Top Controls Bar
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

# Swell Parameter Sidebar
with st.sidebar:
    st.header("Groundswell Configuration")
    period_min = st.number_input("Min Groundswell Period (s)", min_value=10.0, max_value=25.0, value=14.0, step=1.0)
    period_max = st.number_input("Max Groundswell Period (s)", min_value=12.0, max_value=30.0, value=22.0, step=1.0)
    dir_min = st.number_input("Brooks Window Min (° True)", 0, 360, 190)
    dir_max = st.number_input("Brooks Window Max (° True)", 0, 360, 220)

    st.header("Windswell Configuration")
    wind_p_min = st.number_input("Min Windchop Period (s)", min_value=2.0, max_value=8.0, value=4.0, step=0.5)
    wind_p_max = st.number_input("Max Windchop Period (s)", min_value=4.0, max_value=12.0, value=8.0, step=0.5)

@st.cache_data(ttl=900)
def fetch_and_process_cdip(station, p_min, p_max, d_min, d_max, wp_min, wp_max):
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

    # Clean fill/missing values
    fill_mask = (z_raw < -900) | (x_raw < -900) | (y_raw < -900)
    z_raw[fill_mask] = 0.0
    x_raw[fill_mask] = 0.0
    y_raw[fill_mask] = 0.0

    n_pts = len(z_raw)
    time_min = np.arange(n_pts) / (eff_fs * 60.0)

    # ------------------ 1. GROUNDSWELL EXTRACTION ------------------
    b_gs, a_gs = butter(4, [1.0 / p_max, 1.0 / p_min], btype="band", fs=eff_fs)
    z_gs = filtfilt(b_gs, a_gs, z_raw)
    x_gs = filtfilt(b_gs, a_gs, x_raw)
    y_gs = filtfilt(b_gs, a_gs, y_raw)

    z_env_gs = np.abs(hilbert(z_gs))
    
    # Envelope visual smoothing
    smooth_win = int(eff_fs * 8.0)
    if smooth_win > 1:
        kernel = np.hanning(smooth_win)
        kernel /= np.sum(kernel)
        z_env_gs_smooth = np.convolve(z_env_gs, kernel, mode='same')
    else:
        z_env_gs_smooth = z_env_gs

    # Groundswell Set Detection
    min_dist_gs = int(70.0 * eff_fs)
    thresh_gs = float(np.mean(z_env_gs) + 0.75 * np.std(z_env_gs))
    peaks_gs, _ = find_peaks(z_env_gs, height=thresh_gs, distance=min_dist_gs)

    gs_packets = []
    for p in peaks_gs:
        w = int(25.0 * eff_fs)
        idx_s = max(0, p - w)
        idx_e = min(n_pts, p + w)
        
        sub_peaks, _ = find_peaks(z_gs[idx_s:idx_e], distance=int(eff_fs * p_min * 0.7))
        wave_count = max(len(sub_peaks), 1)

        packet_height = (np.max(z_gs[idx_s:idx_e]) - np.min(z_gs[idx_s:idx_e])) * 3.28084
        dx = np.mean(x_gs[idx_s:idx_e])
        dy = np.mean(y_gs[idx_s:idx_e])
        angle_deg = (np.degrees(np.arctan2(-dy, -dx)) + 360.0) % 360.0
        
        if d_min <= d_max:
            is_valid = d_min <= angle_deg <= d_max
        else:
            is_valid = angle_deg >= d_min or angle_deg <= d_max

        gs_packets.append({
            "index": p,
            "time_min": time_min[p],
            "height_ft": packet_height,
            "direction": angle_deg,
            "waves": wave_count,
            "valid": is_valid
        })

    # ------------------ 2. WINDSWELL EXTRACTION ------------------
    # Nyquist limit check for subsampled rate (~0.32 Hz)
    nyq = eff_fs / 2.0
    high_wind = min(1.0 / wp_min, nyq * 0.95)
    low_wind = 1.0 / wp_max
    
    b_ws, a_ws = butter(3, [low_wind, high_wind], btype="band", fs=eff_fs)
    z_ws = filtfilt(b_ws, a_ws, z_raw)
    x_ws = filtfilt(b_ws, a_ws, x_raw)
    y_ws = filtfilt(b_ws, a_ws, y_raw)

    z_env_ws = np.abs(hilbert(z_ws))
    thresh_ws = float(np.mean(z_env_ws) + 0.6 * np.std(z_env_ws))
    min_dist_ws = int(20.0 * eff_fs)
    peaks_ws, _ = find_peaks(z_env_ws, height=thresh_ws, distance=min_dist_ws)

    ws_heights = []
    ws_dirs = []
    for p in peaks_ws:
        w = int(6.0 * eff_fs)
        idx_s = max(0, p - w)
        idx_e = min(n_pts, p + w)
        h = (np.max(z_ws[idx_s:idx_e]) - np.min(z_ws[idx_s:idx_e])) * 3.28084
        dx = np.mean(x_ws[idx_s:idx_e])
        dy = np.mean(y_ws[idx_s:idx_e])
        angle = (np.degrees(np.arctan2(-dy, -dx)) + 360.0) % 360.0
        ws_heights.append(h)
        ws_dirs.append(angle)

    if len(peaks_ws) > 1:
        ws_intervals = [time_min[peaks_ws[i+1]] - time_min[peaks_ws[i]] for i in range(len(peaks_ws)-1)]
        avg_ws_interval = np.mean(ws_intervals)
    else:
        avg_ws_interval = 0.0

    wind_summary = {
        "avg_height_ft": np.mean(ws_heights) if ws_heights else 0.0,
        "max_height_ft": np.max(ws_heights) if ws_heights else 0.0,
        "avg_direction": np.mean(ws_dirs) if ws_dirs else 0.0,
        "avg_interval_min": avg_ws_interval,
        "pulse_count": len(peaks_ws)
    }

    return {
        "time_min": time_min,
        "z_filt": z_gs * 3.28084,
        "z_env_smooth": z_env_gs_smooth * 3.28084,
        "threshold": thresh_gs * 3.28084,
        "gs_packets": gs_packets,
        "wind_summary": wind_summary
    }, None

# UI Header
st.title("🏄 LB Surf Consistency Dashboard")
st.caption(f"Real-Time Directional Swell Separation | Analyzing Station {station_id}")

with st.spinner("Processing 4-hour 3D wave displacement..."):
    data, err = fetch_and_process_cdip(station_id, period_min, period_max, dir_min, dir_max, wind_p_min, wind_p_max)

if err:
    st.error(err)
else:
    all_packets = data["gs_packets"]
    valid_packets = [p for p in all_packets if p["valid"]]
    wind = data["wind_summary"]

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

    # ------------------ TOP SECTION: PRIMARY METRIC TILES ------------------
    st.subheader("🎯 Primary Groundswell (Target Window)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Set Lull", f"{avg_lull:.1f} min" if avg_lull > 0 else "N/A")
    m2.metric("Lull Range (Min / Max)", f"{min_lull:.1f} / {max_lull:.1f} min" if avg_lull > 0 else "N/A")
    m3.metric("Deepwater Set Height", f"{avg_set_height:.1f} ft @ {avg_dir:.0f}°" if avg_set_height > 0 else "N/A")
    m4.metric("Sets Detected", f"{len(valid_packets)} sets", delta="Past 4.0 Hours", delta_color="normal")

    # ------------------ BACKGROUND WINDSWELL METRICS ------------------
    st.subheader("💨 Background Windswell Chop Indicator")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Chop Pulse Interval", f"{wind['avg_interval_min']:.1f} min" if wind['avg_interval_min'] > 0 else "Continuous")
    w2.metric("Average Chop Height", f"{wind['avg_height_ft']:.1f} ft")
    w3.metric("Peak Chop Spike", f"{wind['max_height_ft']:.1f} ft")
    w4.metric("Mean Chop Direction", f"{wind['avg_direction']:.0f}° True")

    # ------------------ DETAILED LOG TABLE (MOVED UP) ------------------
    if valid_packets:
        st.subheader("📋 Groundswell Set Log")
        rows = []
        for i, p in enumerate(valid_packets):
            wait = f"{(p['time_min'] - valid_packets[i-1]['time_min']):.1f} min" if i > 0 else "—"
            rows.append({
                "Set #": i + 1,
                "Arrival Time": f"+{p['time_min']:.1f} min",
                "Lull Wait": wait,
                "Offshore Height": f"{p['height_ft']:.2f} ft",
                "Waves in Packet": f"~{p['waves']} waves",
                "Direction": f"{p['direction']:.1f}° True"
            })
        st.table(rows)
    else:
        st.info("No groundswell sets crossed the threshold within your directional window over the last 4 hours.")

    # ------------------ BOTTOM SECTION: SMOOTHED VISUAL TRACE ------------------
    st.subheader("📈 Time-Series Waveform & Envelope Analysis")
    fig = go.Figure()

    # Heave displacement line (low opacity)
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_filt"], 
        mode="lines", 
        name=f"Filtered Heave ({period_min:.0f}-{period_max:.0f}s)",
        line=dict(color="rgba(41, 182, 246, 0.35)", width=1.0),
        hoverinfo="skip"
    ))

    # Smoothed Envelope Line
    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_env_smooth"], 
        mode="lines", 
        name="Smoothed Wave Envelope",
        line=dict(color="#ff9800", width=2.0)
    ))

    # Set Trigger Line
    fig.add_hline(
        y=data["threshold"], 
        line=dict(color="#ef5350", dash="dot", width=1.2), 
        annotation_text="Set Trigger Threshold",
        annotation_position="bottom left"
    )

    # Valid Groundswell Sets Markers
    if valid_packets:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in valid_packets],
            y=[p["height_ft"]/2.0 for p in valid_packets],
            mode="markers",
            name="Target Groundswell Set",
            marker=dict(color="#00e676", size=10, symbol="diamond", line=dict(width=1, color="#ffffff")),
            hovertemplate="<b>Set Packet</b><br>Time: +%{x:.1f} min<br>Height: %{customdata[0]:.2f} ft<br>Waves: ~%{customdata[1]} waves<br>Bearing: %{customdata[2]:.1f}° True<extra></extra>",
            customdata=[[p["height_ft"], p["waves"], p["direction"]] for p in valid_packets]
        ))

    # Legend at the very bottom, clean margins
    fig.update_layout(
        xaxis_title="Elapsed Time (Minutes)",
        yaxis_title="Surface Heave (Feet)",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=20, b=10),
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
