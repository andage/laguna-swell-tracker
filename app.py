import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks, welch
import plotly.graph_objects as go
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="lb surf", page_icon="🏄", layout="wide")

# Condensed Mobile-Friendly Styling
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 1.5rem !important; }
    h1 { font-size: 1.4rem !important; margin: 0 0 0.2rem 0 !important; }
    h2, h3 { font-size: 1.05rem !important; margin: 0.4rem 0 0.2rem 0 !important; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 6px 10px !important;
        margin-bottom: 4px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #8b949e !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #f0f6fc !important;
    }
    .stTable { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# Station Catalog
STATIONS = {
    "213 - San Pedro South": {"id": "213", "name": "San Pedro South", "lat": 33.584, "lon": -118.240},
    "092 - San Pedro South (Legacy)": {"id": "092", "name": "San Pedro South (Legacy)", "lat": 33.618, "lon": -118.317},
    "045 - Oceanside Offshore": {"id": "045", "name": "Oceanside Offshore", "lat": 33.178, "lon": -117.472},
    "220 - Mission Bay West": {"id": "220", "name": "Mission Bay West", "lat": 32.749, "lon": -117.378},
    "100 - Torrey Pines Outer": {"id": "100", "name": "Torrey Pines Outer", "lat": 32.930, "lon": -117.392},
    "153 - Imperial Beach Nearshore": {"id": "153", "name": "Imperial Beach Nearshore", "lat": 32.580, "lon": -117.168},
    "241 - Del Mar Nearshore": {"id": "241", "name": "Del Mar Nearshore", "lat": 32.956, "lon": -117.279},
    "028 - San Pedro": {"id": "028", "name": "San Pedro (Outer Shelf)", "lat": 33.564, "lon": -118.477},
    "215 - Santa Monica Bay": {"id": "215", "name": "Santa Monica Bay", "lat": 33.855, "lon": -118.634},
    "111 - San Pedro Channel": {"id": "111", "name": "San Pedro Channel", "lat": 33.606, "lon": -118.318},
    "067 - San Nicolas Island": {"id": "067", "name": "San Nicolas Island Outer", "lat": 33.221, "lon": -119.881}
}

# ------------------ TOP CONTROLS ------------------
st.title("🏄 lb surf")

c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    selected_label = st.selectbox("Buoy Station", list(STATIONS.keys()), index=0, label_visibility="collapsed")
    station_info = STATIONS[selected_label]
    station_id = station_info["id"]
    station_name = station_info["name"]
with c_top2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Time Window Selection
c_mode, c_date, c_time = st.columns([1.5, 1.2, 1])
with c_mode:
    time_mode = st.selectbox("Time Mode", ["Live (Latest 4h)", "Historical Lookback"], label_visibility="collapsed")

selected_end_epoch = None
if time_mode == "Historical Lookback":
    with c_date:
        default_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        target_date = st.date_input("Date", value=default_date, label_visibility="collapsed")
    with c_time:
        target_time = st.time_input("Time", value=time(12, 0), label_visibility="collapsed")
    dt_combined = datetime.combine(target_date, target_time).replace(tzinfo=timezone.utc)
    selected_end_epoch = int(dt_combined.timestamp())

@st.cache_data(ttl=900)
def fetch_and_analyze(station, end_epoch):
    hours = 4.0
    rt_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_rt.nc"
    xy_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    try:
        ds_rt = xr.open_dataset(rt_url, decode_times=False)
        ds_xy = xr.open_dataset(xy_url, decode_times=False)
    except Exception as e:
        return None, f"CDIP connection failed for Station {station}: {e}"

    if "xyzZDisplacement" not in ds_xy or len(ds_xy.xyzZDisplacement) == 0:
        return None, f"Station {station} displacement stream currently offline."

    # Pull Official CDIP Dominant Period
    try:
        cdip_tp = float(ds_rt.waveTp[-1].values)
    except Exception:
        cdip_tp = 14.0 # Fallback

    try:
        raw_fs = ds_xy.xyzSampleRate.values.item() if hasattr(ds_xy.xyzSampleRate.values, "item") else ds_xy.xyzSampleRate.values
        fs = float(raw_fs)
    except Exception:
        fs = 1.28

    stride = 2
    eff_fs = fs / stride
    samples_needed = int(hours * 3600 * fs)
    total_len = len(ds_xy.xyzZDisplacement)

    start_time_base = None
    if "xyzStartTime" in ds_xy:
        try:
            val = float(ds_xy.xyzStartTime.values)
            if 946684800 <= val <= 2051222400:
                start_time_base = val
        except Exception:
            pass

    if start_time_base is not None:
        file_end_epoch = start_time_base + (total_len / fs)
    else:
        file_end_epoch = datetime.now(timezone.utc).timestamp()
        start_time_base = file_end_epoch - (total_len / fs)

    if end_epoch is None:
        idx_start = max(0, total_len - samples_needed)
        idx_end = total_len
    else:
        target_idx = int((end_epoch - start_time_base) * fs)
        idx_end = min(total_len, max(samples_needed, target_idx))
        idx_start = max(0, idx_end - samples_needed)

    try:
        z_raw = ds_xy.xyzZDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
        x_raw = ds_xy.xyzXDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
        y_raw = ds_xy.xyzYDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
    except Exception as e:
        return None, f"Data slicing error: {e}"

    fill_mask = (z_raw < -900) | (x_raw < -900) | (y_raw < -900)
    z_raw[fill_mask] = 0.0
    x_raw[fill_mask] = 0.0
    y_raw[fill_mask] = 0.0

    n_pts = len(z_raw)
    time_min = np.arange(n_pts) / (eff_fs * 60.0)

    # Spectral Analysis for Secondary Period
    freqs, psd = welch(z_raw, fs=eff_fs, nperseg=min(len(z_raw), 1024))
    valid_mask = (freqs >= 0.038) & (freqs <= 0.28)
    f_band = freqs[valid_mask]
    psd_band = psd[valid_mask]

    peaks, _ = find_peaks(psd_band, distance=int(0.025 / (freqs[1] - freqs[0])))
    
    f_dom = 1.0 / cdip_tp
    f_sec = 0.16 # Fallback 6.2s
    if len(peaks) > 0:
        sorted_p = peaks[np.argsort(psd_band[peaks])[::-1]]
        for p_idx in sorted_p:
            peak_f = f_band[p_idx]
            # Find the strongest spectral peak that is significantly different from the CDIP Tp
            if abs(peak_f - f_dom) > 0.03:
                f_sec = peak_f
                break

    t_sec = 1.0 / f_sec

    # 2. Decomposition Function
    def decompose_band(f_center, bw=0.022):
        low = max(0.035, f_center - bw)
        high = min((eff_fs / 2.0) * 0.95, f_center + bw)
        b, a = butter(3, [low, high], btype="band", fs=eff_fs)
        z_f = filtfilt(b, a, z_raw)
        x_f = filtfilt(b, a, x_raw)
        y_f = filtfilt(b, a, y_raw)
        env = np.abs(hilbert(z_f))
        
        sm_len = int(eff_fs * 8.0)
        kernel = np.hanning(sm_len)
        kernel /= np.sum(kernel)
        env_sm = np.convolve(env, kernel, mode="same")

        thresh = float(np.mean(env) + 0.7 * np.std(env))
        min_d = int((1.0 / f_center) * 3.5 * eff_fs)
        p_idx, _ = find_peaks(env, height=thresh, distance=min_d)

        pkts = []
        for p in p_idx:
            w = int((1.0 / f_center) * 1.5 * eff_fs)
            s_i, e_i = max(0, p - w), min(n_pts, p + w)
            h = (np.max(z_f[s_i:e_i]) - np.min(z_f[s_i:e_i])) * 3.28084
            dx = np.mean(x_f[s_i:e_i])
            dy = np.mean(y_f[s_i:e_i])
            deg = (np.degrees(np.arctan2(-dy, -dx)) + 360.0) % 360.0
            
            sub_p, _ = find_peaks(z_f[s_i:e_i], distance=int(eff_fs * (1.0 / f_center) * 0.7))
            waves = max(len(sub_p), 1)

            pkts.append({"time_min": time_min[p], "height_ft": h, "dir": deg, "waves": waves})

        return z_f * 3.28084, env_sm * 3.28084, thresh * 3.28084, pkts

    z_dom_f, env_dom, thresh_dom, pkts_dom = decompose_band(f_dom, bw=0.025)
    z_sec_f, env_sec, thresh_sec, pkts_sec = decompose_band(f_sec, bw=0.035)

    return {
        "time_min": time_min,
        "t_dom": cdip_tp,
        "t_sec": t_sec,
        "z_dom_f": z_dom_f,
        "env_dom": env_dom,
        "thresh_dom": thresh_dom,
        "pkts_dom": pkts_dom,
        "pkts_sec": pkts_sec
    }, None

with st.spinner("Processing dual-wave decomposition..."):
    data, err = fetch_and_analyze(station_id, selected_end_epoch)

if err:
    st.error(err)
else:
    # Dominant Wave Metrics
    p_dom = data["pkts_dom"]
    if len(p_dom) > 1:
        lulls_dom = [p_dom[i+1]["time_min"] - p_dom[i]["time_min"] for i in range(len(p_dom)-1)]
        avg_lull = np.mean(lulls_dom)
        lull_str = f"{np.min(lulls_dom):.0f} / {np.max(lulls_dom):.0f}m"
        h_dom = np.mean([p["height_ft"] for p in p_dom])
        d_dom = np.mean([p["dir"] for p in p_dom])
    elif len(p_dom) == 1:
        avg_lull, lull_str = 0.0, "—"
        h_dom, d_dom = p_dom[0]["height_ft"], p_dom[0]["dir"]
    else:
        avg_lull, lull_str, h_dom, d_dom = 0.0, "—", 0.0, 168.0

    # Secondary Wave Metrics
    p_sec = data["pkts_sec"]
    h_sec = np.mean([p["height_ft"] for p in p_sec]) if p_sec else 0.0
    d_sec = np.mean([p["dir"] for p in p_sec]) if p_sec else 280.0
    sec_type = "Groundswell" if data["t_sec"] >= 10.0 else "Windchop"

    # 1. Condensed Primary Metrics Row
    st.markdown(f"**Dominant Swell ({data['t_dom']:.0f}s Component)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Set Lull (Avg)", f"{avg_lull:.1f} min" if avg_lull > 0 else "—")
    c2.metric("Lull Range", lull_str)
    c3.metric("Deepwater Set", f"{h_dom:.1f}ft @ {d_dom:.0f}°" if h_dom > 0 else "—")
    c4.metric("Sets (4h)", f"{len(p_dom)} sets")

    # 2. Condensed Secondary Metrics Row
    st.markdown(f"**Secondary Wave ({data['t_sec']:.0f}s {sec_type})**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pulse Spacing", f"{(240.0 / len(p_sec)):.1f} min" if len(p_sec) > 1 else "Continuous")
    s2.metric("Component Ht", f"{h_sec:.1f} ft")
    s3.metric("Direction", f"{d_sec:.0f}° True")
    s4.metric("Band Peak", f"{data['t_sec']:.1f} sec")

    # 3. Compact Set Table
    if p_dom:
        rows = []
        for i, p in enumerate(p_dom):
            wait = f"{(p['time_min'] - p_dom[i-1]['time_min']):.0f}m" if i > 0 else "—"
            rows.append({
                "Set": i + 1, "Time": f"+{p['time_min']:.0f}m", "Lull": wait,
                "Set Ht": f"{p['height_ft']:.1f}ft", "Waves": f"~{p['waves']}", "Dir": f"{p['dir']:.0f}°"
            })
        st.table(rows[:8]) # Display top 8 sets in compact format

    # 4. Waveform & Envelope Analysis
    st.markdown("**Dominant Wave Groups & Hilbert Envelope**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time_min"], y=data["z_dom_f"], mode="lines", name=f"{data['t_dom']:.0f}s Heave",
        line=dict(color="rgba(41, 182, 246, 0.35)", width=1), hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=data["time_min"], y=data["env_dom"], mode="lines", name="Envelope",
        line=dict(color="#ff9800", width=1.8)
    ))
    fig.add_hline(y=data["thresh_dom"], line=dict(color="#ef5350", dash="dot", width=1))

    if p_dom:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in p_dom], y=[p["height_ft"]/2.0 for p in p_dom],
            mode="markers", name="Set",
            marker=dict(color="#00e676", size=8, symbol="diamond")
        ))

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=10))
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Bulletproof Southern California Swell Shadowing Map
    st.markdown(f"**SoCal Swell Shadow Projection ({d_dom:.0f}° True)**")
    
    map_fig = go.Figure()

    # SoCal Coastline Geometry
    coast_lat = [34.45, 34.41, 34.27, 34.02, 33.74, 33.74, 33.60, 33.535, 33.46, 33.19, 32.93, 32.67, 32.55]
    coast_lon = [-120.47, -119.69, -119.29, -118.80, -118.40, -118.11, -117.88, -117.78, -117.70, -117.38, -117.26, -117.24, -117.13]
    map_fig.add_trace(go.Scatter(
        x=coast_lon, y=coast_lat, mode="lines", line=dict(color="#78909c", width=2),
        name="Coastline", hoverinfo="skip"
    ))

    # Channel Islands Geometry
    islands = {
        "Catalina": [(33.48, -118.60), (33.43, -118.50), (33.32, -118.32), (33.30, -118.35), (33.35, -118.52), (33.48, -118.60)],
        "San Clemente": [(33.03, -118.60), (32.95, -118.55), (32.81, -118.36), (32.82, -118.42), (33.00, -118.62), (33.03, -118.60)],
        "San Nicolas": [(33.28, -119.58), (33.25, -119.45), (33.22, -119.48), (33.25, -119.59), (33.28, -119.58)]
    }

    # Shadow vectors downwave: traveling towards (d_dom + 180°)
    sh_rad = np.radians((d_dom + 180.0) % 360.0)
    sh_len = 1.15
    d_lat = sh_len * np.cos(sh_rad)
    d_lon = sh_len * np.sin(sh_rad)

    # Project Shadow Cones
    for name, pts in islands.items():
        sh_x = [p[1] for p in pts] + [p[1] + d_lon for p in reversed(pts)]
        sh_y = [p[0] for p in pts] + [p[0] + d_lat for p in reversed(pts)]
        map_fig.add_trace(go.Scatter(
            x=sh_x, y=sh_y, fill="toself", fillcolor="rgba(239, 83, 80, 0.3)",
            line=dict(color="rgba(239, 83, 80, 0.4)", width=1), name=f"{name} Shadow", hoverinfo="skip"
        ))

    # Draw Islands
    for name, pts in islands.items():
        map_fig.add_trace(go.Scatter(
            x=[p[1] for p in pts], y=[p[0] for p in pts], fill="toself", fillcolor="#455a64",
            line=dict(color="#90a4ae", width=1.5), name=name, hoverinfo="text", text=name
        ))

    # Brooks Street
    map_fig.add_trace(go.Scatter(
        x=[-117.778], y=[33.535], mode="markers+text",
        marker=dict(size=10, color="#00e676", symbol="star"),
        text=["Brooks St"], textposition="top right", name="Brooks St"
    ))

    # Selected Buoy
    map_fig.add_trace(go.Scatter(
        x=[station_info["lon"]], y=[station_info["lat"]], mode="markers+text",
        marker=dict(size=8, color="#29b6f6", symbol="circle"),
        text=[f"Buoy {station_id}"], textposition="bottom center", name=f"Buoy {station_id}"
    ))

    # Swell Arrow Indicator
    arr_x = [-118.8, -118.8 + 0.35 * np.sin(sh_rad)]
    arr_y = [32.5, 32.5 + 0.35 * np.cos(sh_rad)]
    map_fig.add_trace(go.Scatter(
        x=arr_x, y=arr_y, mode="lines+markers",
        line=dict(color="#00e676", width=2.5), marker=dict(size=[0, 8], symbol="triangle-up"),
        name=f"Swell Track ({d_dom:.0f}°)"
    ))

    map_fig.update_layout(
        template="plotly_dark", height=380,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        xaxis=dict(range=[-120.2, -117.0], showgrid=True, gridcolor="#21262d", zeroline=False),
        yaxis=dict(range=[32.4, 34.6], showgrid=True, gridcolor="#21262d", scaleanchor="x", scaleratio=1.19, zeroline=False),
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9))
    )
    st.plotly_chart(map_fig, use_container_width=True)
