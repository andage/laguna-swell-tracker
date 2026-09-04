import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="lb surf", page_icon="🏄", layout="wide")

# Active Southern California Stations (Realtime _xy.nc streams)
STATIONS = {
    "092 - San Pedro South": {"id": "092", "name": "San Pedro South"},
    "045 - Oceanside Offshore": {"id": "045", "name": "Oceanside Offshore"},
    "220 - Mission Bay West": {"id": "220", "name": "Mission Bay West"},
    "100 - Torrey Pines Outer": {"id": "100", "name": "Torrey Pines Outer"},
    "153 - Imperial Beach Nearshore": {"id": "153", "name": "Imperial Beach Nearshore"},
    "241 - Del Mar Nearshore": {"id": "241", "name": "Del Mar Nearshore"},
    "028 - San Pedro": {"id": "028", "name": "San Pedro (Outer Shelf)"},
    "215 - Santa Monica Bay": {"id": "215", "name": "Santa Monica Bay"},
    "111 - San Pedro Channel": {"id": "111", "name": "San Pedro Channel"},
    "067 - San Nicolas Island": {"id": "067", "name": "San Nicolas Island Outer"},
    "222 - San Pedro South Shelf": {"id": "222", "name": "San Pedro South Shelf"}
}

# Top Station Selection & Controls
col_top1, col_top2 = st.columns([3, 1])
with col_top1:
    selected_label = st.selectbox("Select Station", list(STATIONS.keys()), index=0)
    station_info = STATIONS[selected_label]
    station_id = station_info["id"]
    station_name = station_info["name"]
with col_top2:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# Sidebar: Time Window & Swell Bands
with st.sidebar:
    st.header("🕒 Time Window Selection")
    time_mode = st.radio("Mode", ["Live (Latest 4 Hours)", "Historical Lookback"], index=0)
    
    selected_end_epoch = None
    if time_mode == "Historical Lookback":
        default_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        target_date = st.date_input("Target Date (UTC)", value=default_date)
        target_time = st.time_input("Target End Time (UTC)", value=time(12, 0))
        dt_combined = datetime.combine(target_date, target_time).replace(tzinfo=timezone.utc)
        selected_end_epoch = int(dt_combined.timestamp())
        st.caption(f"Window: 4h prior to {dt_combined.strftime('%Y-%m-%d %H:%M UTC')}")

    st.header("🎯 Groundswell Filter")
    period_min = st.number_input("Min Groundswell Period (s)", 10.0, 25.0, 14.0, 1.0)
    period_max = st.number_input("Max Groundswell Period (s)", 12.0, 30.0, 22.0, 1.0)
    dir_min = st.number_input("Brooks Window Min (° True)", 0, 360, 190)
    dir_max = st.number_input("Brooks Window Max (° True)", 0, 360, 220)

    st.header("💨 Windswell Filter")
    wind_p_min = st.number_input("Min Windchop Period (s)", 2.0, 8.0, 4.0, 0.5)
    wind_p_max = st.number_input("Max Windchop Period (s)", 4.0, 12.0, 8.0, 0.5)

@st.cache_data(ttl=900)
def fetch_and_process_cdip(station, end_epoch, p_min, p_max, d_min, d_max, wp_min, wp_max):
    hours = 4.0
    url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    try:
        ds = xr.open_dataset(url)
    except Exception as e:
        return None, f"Failed to connect to CDIP OpenDAP endpoint for Station {station}: {e}"

    fs = float(ds.xyzSampleRate.values)
    stride = 2
    eff_fs = fs / stride
    samples_needed = int(hours * 3600 * fs)

    total_len = len(ds.xyzZDisplacement)
    
    if end_epoch is None:
        idx_start = max(0, total_len - samples_needed)
        idx_end = total_len
    else:
        start_time_base = int(ds.xyzStartTime.values)
        target_sample_index = int((end_epoch - start_time_base) * fs)
        
        if target_sample_index <= samples_needed:
            return None, "Target historical time is prior to the start of this file's recorded telemetry."
        if target_sample_index > total_len:
            target_sample_index = total_len
            
        idx_end = target_sample_index
        idx_start = max(0, idx_end - samples_needed)

    try:
        z_raw = ds.xyzZDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
        x_raw = ds.xyzXDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
        y_raw = ds.xyzYDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
    except Exception as e:
        return None, f"Error reading displacement arrays: {e}"

    fill_mask = (z_raw < -900) | (x_raw < -900) | (y_raw < -900)
    z_raw[fill_mask] = 0.0
    x_raw[fill_mask] = 0.0
    y_raw[fill_mask] = 0.0

    n_pts = len(z_raw)
    time_min = np.arange(n_pts) / (eff_fs * 60.0)

    # 1. Groundswell Processing
    b_gs, a_gs = butter(4, [1.0 / p_max, 1.0 / p_min], btype="band", fs=eff_fs)
    z_gs = filtfilt(b_gs, a_gs, z_raw)
    x_gs = filtfilt(b_gs, a_gs, x_raw)
    y_gs = filtfilt(b_gs, a_gs, y_raw)

    z_env_gs = np.abs(hilbert(z_gs))
    smooth_win = int(eff_fs * 8.0)
    if smooth_win > 1:
        kernel = np.hanning(smooth_win)
        kernel /= np.sum(kernel)
        z_env_gs_smooth = np.convolve(z_env_gs, kernel, mode='same')
    else:
        z_env_gs_smooth = z_env_gs

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

    # 2. Windswell Processing
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

    ws_heights, ws_dirs = [], []
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
    }

    return {
        "time_min": time_min,
        "z_filt": z_gs * 3.28084,
        "z_env_smooth": z_env_gs_smooth * 3.28084,
        "threshold": thresh_gs * 3.28084,
        "gs_packets": gs_packets,
        "wind_summary": wind_summary
    }, None

# Main Body
st.title("🏄 lb surf")
st.markdown(f"### Currently Monitoring: **Buoy {station_id} — {station_name}**")

with st.spinner(f"Querying 3D wave telemetry from Buoy {station_id} ({station_name})..."):
    data, err = fetch_and_process_cdip(station_id, selected_end_epoch, period_min, period_max, dir_min, dir_max, wind_p_min, wind_p_max)

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
    elif len(valid_packets) == 1:
        avg_lull = min_lull = max_lull = 0.0
        avg_set_height = valid_packets[0]["height_ft"]
        avg_dir = valid_packets[0]["direction"]
    else:
        avg_lull = min_lull = max_lull = avg_set_height = avg_dir = 0.0

    # Top Metric Tiles
    st.subheader(f"🎯 Primary Groundswell — Buoy {station_id} ({station_name})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Set Lull", f"{avg_lull:.1f} min" if avg_lull > 0 else "N/A")
    m2.metric("Lull Range (Min / Max)", f"{min_lull:.1f} / {max_lull:.1f} min" if avg_lull > 0 else "N/A")
    m3.metric("Deepwater Set Height", f"{avg_set_height:.1f} ft @ {avg_dir:.0f}°" if avg_set_height > 0 else "N/A")
    delta_tag = "Historical 4.0h Slice" if time_mode == "Historical Lookback" else "Past 4.0 Hours"
    m4.metric("Sets Detected", f"{len(valid_packets)} sets", delta=delta_tag, delta_color="normal")

    # Windswell Indicator Tiles
    st.subheader(f"💨 Background Windswell Chop — Buoy {station_id}")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Chop Pulse Spacing", f"{wind['avg_interval_min']:.1f} min" if wind['avg_interval_min'] > 0 else "Continuous")
    w2.metric("Average Chop Height", f"{wind['avg_height_ft']:.1f} ft")
    w3.metric("Peak Chop Spike", f"{wind['max_height_ft']:.1f} ft")
    w4.metric("Mean Chop Direction", f"{wind['avg_direction']:.0f}° True")

    # Detailed Set Arrival Table
    if valid_packets:
        st.subheader(f"📋 Groundswell Set Log for Buoy {station_id} ({station_name})")
        rows = []
        for i, p in enumerate(valid_packets):
            wait = f"{(p['time_min'] - valid_packets[i-1]['time_min']):.1f} min" if i > 0 else "—"
            rows.append({
                "Set #": i + 1,
                "Arrival Time": f"+{p['time_min']:.1f} min",
                "Lull Duration": wait,
                "Offshore Height": f"{p['height_ft']:.2f} ft",
                "Waves in Packet": f"~{p['waves']} waves",
                "Direction": f"{p['direction']:.1f}° True"
            })
        st.table(rows)
    else:
        st.info(f"No groundswell sets crossed the threshold within your directional window on Buoy {station_id} during this 4-hour window.")

    # Bottom Heave and Envelope Graph
    st.subheader(f"📈 Waveform & Envelope Analysis — Buoy {station_id} ({station_name})")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_filt"], 
        mode="lines", 
        name=f"Filtered Heave ({period_min:.0f}-{period_max:.0f}s)",
        line=dict(color="rgba(41, 182, 246, 0.35)", width=1.0),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=data["time_min"], 
        y=data["z_env_smooth"], 
        mode="lines", 
        name="Smoothed Wave Envelope",
        line=dict(color="#ff9800", width=2.0)
    ))

    fig.add_hline(
        y=data["threshold"], 
        line=dict(color="#ef5350", dash="dot", width=1.2), 
        annotation_text="Set Threshold",
        annotation_position="bottom left"
    )

    if valid_packets:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in valid_packets],
            y=[p["height_ft"]/2.0 for p in valid_packets],
            mode="markers",
            name=f"Target Set ({station_name})",
            marker=dict(color="#00e676", size=10, symbol="diamond", line=dict(width=1, color="#ffffff")),
            hovertemplate="<b>Set Packet</b><br>Time: +%{x:.1f} min<br>Height: %{customdata[0]:.2f} ft<br>Waves: ~%{customdata[1]} waves<br>Bearing: %{customdata[2]:.1f}° True<extra></extra>",
            customdata=[[p["height_ft"], p["waves"], p["direction"]] for p in valid_packets]
        ))

    fig.update_layout(
        xaxis_title="Elapsed Time in Window (Minutes)",
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
