import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="lb surf", page_icon="🏄", layout="wide")

# Active Southern California Stations
STATIONS = {
    "067 - San Nicolas Island": {"id": "067", "name": "San Nicolas Island Outer", "lat": 33.221, "lon": -119.881},
    "092 - San Pedro South": {"id": "092", "name": "San Pedro South", "lat": 33.618, "lon": -118.317},
    "045 - Oceanside Offshore": {"id": "045", "name": "Oceanside Offshore", "lat": 33.178, "lon": -117.472},
    "220 - Mission Bay West": {"id": "220", "name": "Mission Bay West", "lat": 32.749, "lon": -117.378},
    "100 - Torrey Pines Outer": {"id": "100", "name": "Torrey Pines Outer", "lat": 32.930, "lon": -117.392},
    "153 - Imperial Beach Nearshore": {"id": "153", "name": "Imperial Beach Nearshore", "lat": 32.580, "lon": -117.168},
    "241 - Del Mar Nearshore": {"id": "241", "name": "Del Mar Nearshore", "lat": 32.956, "lon": -117.279},
    "028 - San Pedro": {"id": "028", "name": "San Pedro (Outer Shelf)", "lat": 33.564, "lon": -118.477},
    "215 - Santa Monica Bay": {"id": "215", "name": "Santa Monica Bay", "lat": 33.855, "lon": -118.634},
    "111 - San Pedro Channel": {"id": "111", "name": "San Pedro Channel", "lat": 33.606, "lon": -118.318},
    "222 - San Pedro South Shelf": {"id": "222", "name": "San Pedro South Shelf", "lat": 33.618, "lon": -118.317}
}

# ------------------ TOP CONTROLS ------------------
st.title("🏄 lb surf")

c_top1, c_top2 = st.columns([2, 1])
with c_top1:
    selected_label = st.selectbox("Select Buoy Station", list(STATIONS.keys()), index=0)
    station_info = STATIONS[selected_label]
    station_id = station_info["id"]
    station_name = station_info["name"]
with c_top2:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Data / Clear Cache"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"### Currently Monitoring: **Buoy {station_id} — {station_name}**")

time_mode = st.selectbox("Time Window Mode", ["Live (Latest 4 Hours)", "Historical Lookback"], index=0)

selected_end_epoch = None
if time_mode == "Historical Lookback":
    c_hist1, c_hist2 = st.columns(2)
    with c_hist1:
        # Default to 2 days ago to guarantee buffer hit
        default_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        target_date = st.date_input("Target Date (UTC)", value=default_date)
    with c_hist2:
        target_time = st.time_input("Target End Time (UTC)", value=time(12, 0))
    dt_combined = datetime.combine(target_date, target_time).replace(tzinfo=timezone.utc)
    selected_end_epoch = int(dt_combined.timestamp())
    st.info(f"Targeting window ending at: **{dt_combined.strftime('%Y-%m-%d %H:%M UTC')}**")

# Swell Filter Settings
with st.sidebar:
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
    rt_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    # Check realtime displacement buffer
    try:
        ds = xr.open_dataset(rt_url)
    except Exception as e:
        return None, f"Could not connect to CDIP real-time endpoint for Station {station}: {e}"

    fs = float(ds.xyzSampleRate.values) if "xyzSampleRate" in ds else 1.28
    stride = 2
    eff_fs = fs / stride
    samples_needed = int(hours * 3600 * fs)
    total_len = len(ds.xyzZDisplacement)

    start_time_base = float(ds.xyzStartTime.values) if "xyzStartTime" in ds else None
    if start_time_base is not None:
        file_end_epoch = start_time_base + (total_len / fs)
        dt_start_str = datetime.fromtimestamp(start_time_base, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        dt_end_str = datetime.fromtimestamp(file_end_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        range_desc = f"{dt_start_str} to {dt_end_str}"
    else:
        file_end_epoch = datetime.now(timezone.utc).timestamp()
        range_desc = "Recent 3-5 days buffer"

    # --- Case 1: Real-Time or within Buffer ---
    within_buffer = False
    if end_epoch is None:
        within_buffer = True
        idx_start = max(0, total_len - samples_needed)
        idx_end = total_len
    elif start_time_base is not None and (start_time_base + samples_needed/fs) <= end_epoch <= (file_end_epoch + 3600):
        within_buffer = True
        target_sample_index = int((end_epoch - start_time_base) * fs)
        idx_end = min(total_len, target_sample_index)
        idx_start = max(0, idx_end - samples_needed)

    if within_buffer:
        try:
            z_raw = ds.xyzZDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
            x_raw = ds.xyzXDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
            y_raw = ds.xyzYDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
        except Exception as e:
            return None, f"Error slicing displacement arrays: {e}"

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

        ws_intervals = [time_min[peaks_ws[i+1]] - time_min[peaks_ws[i]] for i in range(len(peaks_ws)-1)] if len(peaks_ws) > 1 else []

        return {
            "mode": "displacement",
            "time_min": time_min,
            "z_filt": z_gs * 3.28084,
            "z_env_smooth": z_env_gs_smooth * 3.28084,
            "threshold": thresh_gs * 3.28084,
            "gs_packets": gs_packets,
            "wind_summary": {
                "avg_height_ft": np.mean(ws_heights) if ws_heights else 0.0,
                "max_height_ft": np.max(ws_heights) if ws_heights else 0.0,
                "avg_direction": np.mean(ws_dirs) if ws_dirs else 0.0,
                "avg_interval_min": np.mean(ws_intervals) if ws_intervals else 0.0,
            },
            "file_range": range_desc
        }, None

    # --- Case 2: Deep Historical Archive (Fallback to Processed Spectral Timeseries) ---
    hist_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{station}p1/{station}p1_historic.nc"
    try:
        hds = xr.open_dataset(hist_url)
    except Exception:
        return None, f"Target date is outside the 3-day raw buffer (**{range_desc}**), and historical archive file is unavailable."

    # Look up nearest waveTime index
    wave_times = hds.waveTime.values
    t_idx = np.argmin(np.abs(wave_times - end_epoch))
    
    closest_epoch = wave_times[t_idx]
    if abs(closest_epoch - end_epoch) > 3600 * 24 * 7: # Over 7 days mismatch
        return None, f"Target date could not be found in historical archive for Station {station}."

    hs = float(hds.waveHs[t_idx].values) * 3.28084
    tp = float(hds.waveTp[t_idx].values)
    dp = float(hds.waveDp[t_idx].values)

    return {
        "mode": "archive_spectral",
        "hs_ft": hs,
        "tp_s": tp,
        "dp_deg": dp,
        "target_dt": datetime.fromtimestamp(closest_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        "buffer_span": range_desc
    }, None

with st.spinner(f"Querying wave telemetry from Buoy {station_id}..."):
    data, err = fetch_and_process_cdip(station_id, selected_end_epoch, period_min, period_max, dir_min, dir_max, wind_p_min, wind_p_max)

if err:
    st.error(err)
else:
    if data["mode"] == "displacement":
        all_packets = data["gs_packets"]
        valid_packets = [p for p in all_packets if p["valid"]]
        wind = data["wind_summary"]
        st.caption(f"Active Real-Time Buffer Span: **{data['file_range']}**")

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
            avg_lull = min_lull = max_lull = avg_set_height = 0.0
            avg_dir = 205.0

        # 1. Primary Metrics
        st.subheader(f"🎯 Primary Groundswell — Buoy {station_id} ({station_name})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average Set Lull", f"{avg_lull:.1f} min" if avg_lull > 0 else "N/A")
        m2.metric("Lull Range (Min / Max)", f"{min_lull:.1f} / {max_lull:.1f} min" if avg_lull > 0 else "N/A")
        m3.metric("Deepwater Set Height", f"{avg_set_height:.1f} ft @ {avg_dir:.0f}°" if avg_set_height > 0 else "N/A")
        delta_tag = "Historical Buffer Slice" if time_mode == "Historical Lookback" else "Past 4.0 Hours"
        m4.metric("Sets Detected", f"{len(valid_packets)} sets", delta=delta_tag, delta_color="normal")

        # 2. Windswell Metrics
        st.subheader(f"💨 Background Windswell Chop — Buoy {station_id}")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Chop Pulse Spacing", f"{wind['avg_interval_min']:.1f} min" if wind['avg_interval_min'] > 0 else "Continuous")
        w2.metric("Average Chop Height", f"{wind['avg_height_ft']:.1f} ft")
        w3.metric("Peak Chop Spike", f"{wind['max_height_ft']:.1f} ft")
        w4.metric("Mean Chop Direction", f"{wind['avg_direction']:.0f}° True")

        # 3. Groundswell Set Log Table
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

        # 4. Waveform & Envelope Analysis Plot
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
            height=420,
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

    else:
        # Archive Spectral Summary Mode (For dates older than ~5 days)
        st.warning(f"ℹ️ Selected date precedes the active 3-5 day raw displacement buffer ({data['buffer_span']}). Showing CDIP Permanent Archive Spectral Records for {data['target_dt']}.")
        
        avg_dir = data["dp_deg"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Significant Wave Height (Hs)", f"{data['hs_ft']:.1f} ft")
        a2.metric("Peak Period (Tp)", f"{data['tp_s']:.1f} s")
        a3.metric("Peak Direction (Dp)", f"{data['dp_deg']:.0f}° True")

    # 5. Southern California Swell Shadowing Map
    st.subheader(f"🗺️ Southern California Swell Shadow Projection ({avg_dir:.0f}° True)")

    islands = {
        "Catalina Island": [
            (33.48, -118.60), (33.43, -118.50), (33.32, -118.32), 
            (33.30, -118.35), (33.35, -118.52), (33.48, -118.60)
        ],
        "San Clemente Island": [
            (33.03, -118.60), (32.95, -118.55), (32.81, -118.36),
            (32.82, -118.42), (33.00, -118.62), (33.03, -118.60)
        ],
        "San Nicolas Island": [
            (33.28, -119.58), (33.25, -119.45), (33.22, -119.48), 
            (33.25, -119.59), (33.28, -119.58)
        ]
    }

    shadow_angle_rad = np.radians((avg_dir - 180.0 + 360.0) % 360.0)
    shadow_length = 0.9
    d_lat = shadow_length * np.cos(shadow_angle_rad)
    d_lon = shadow_length * np.sin(shadow_angle_rad)

    map_fig = go.Figure()

    for name, coords in islands.items():
        sh_lats = [pt[0] for pt in coords] + [pt[0] + d_lat for pt in reversed(coords)]
        sh_lons = [pt[1] for pt in coords] + [pt[1] + d_lon for pt in reversed(coords)]
        map_fig.add_trace(go.Scattergeo(
            lat=sh_lats,
            lon=sh_lons,
            fill="toself",
            fillcolor="rgba(239, 83, 80, 0.28)",
            line=dict(color="rgba(239, 83, 80, 0.45)", width=1),
            name=f"Shadow ({name})",
            hoverinfo="text",
            text=f"Blocked Zone behind {name}"
        ))

    for name, coords in islands.items():
        map_fig.add_trace(go.Scattergeo(
            lat=[pt[0] for pt in coords],
            lon=[pt[1] for pt in coords],
            fill="toself",
            fillcolor="#546e7a",
            line=dict(color="#b0bec5", width=1.5),
            name=name,
            hoverinfo="text",
            text=name
        ))

    brooks_lat, brooks_lon = 33.535, -117.778
    map_fig.add_trace(go.Scattergeo(
        lat=[brooks_lat],
        lon=[brooks_lon],
        mode="markers+text",
        marker=dict(size=11, color="#00e676", symbol="star"),
        text=["Brooks St, Laguna Beach"],
        textposition="top right",
        name="Brooks Street (Laguna)",
        hoverinfo="text"
    ))

    map_fig.add_trace(go.Scattergeo(
        lat=[station_info["lat"]],
        lon=[station_info["lon"]],
        mode="markers+text",
        marker=dict(size=9, color="#29b6f6", symbol="circle"),
        text=[f"Buoy {station_id}"],
        textposition="bottom left",
        name=f"Buoy {station_id} ({station_name})"
    ))

    arrow_lat = [32.4, 32.4 + 0.4 * np.cos(shadow_angle_rad)]
    arrow_lon = [-118.9, -118.9 + 0.4 * np.sin(shadow_angle_rad)]
    map_fig.add_trace(go.Scattergeo(
        lat=arrow_lat,
        lon=arrow_lon,
        mode="lines+markers",
        line=dict(color="#00e676", width=3),
        marker=dict(size=[0, 8], symbol="triangle-up"),
        name=f"Swell Approach Vector ({avg_dir:.0f}°)"
    ))

    map_fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="mercator",
            center=dict(lat=33.35, lon=-118.4),
            lataxis=dict(range=[32.3, 34.1]),
            lonaxis=dict(range=[-120.2, -117.0]),
            showland=True,
            landcolor="#1e222d",
            showocean=True,
            oceancolor="#0f131a",
            showcoastlines=True,
            coastlinecolor="#78909c",
            resolution=50
        ),
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    st.plotly_chart(map_fig, use_container_width=True)
