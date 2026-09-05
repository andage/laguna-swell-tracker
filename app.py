import streamlit as st
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import plotly.graph_objects as go
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="lb surf", page_icon="🏄", layout="wide")

# Ultra-Condensed Mobile CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 1.5rem !important; max-width: 100%; }
    h1 { font-size: 1.3rem !important; margin: 0 0 0.2rem 0 !important; }
    h3 { font-size: 0.95rem !important; margin: 0.2rem 0 0.2rem 0 !important; font-weight: 600; color: #58a6ff; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 4px 8px !important;
        margin-bottom: 2px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #8b949e !important; margin-bottom: 2px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.95rem !important; font-weight: 700 !important; color: #f0f6fc !important; }
    .stTable { font-size: 0.75rem !important; margin-bottom: 0 !important; }
    .stTable th, .stTable td { padding: 4px 6px !important; }
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
    "067 - San Nicolas Island": {"id": "067", "name": "San Nicolas Island Outer", "lat": 33.221, "lon": -119.881}
}

# ------------------ TOP CONTROLS ------------------
st.title("🏄 lb surf")

c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    selected_label = st.selectbox("Select Buoy Station", list(STATIONS.keys()), index=0)
    station_info = STATIONS[selected_label]
    station_id = station_info["id"]
    station_name = station_info["name"]
with c_top2:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"### Currently Monitoring: **Buoy {station_id} — {station_name}**")

c_mode, c_date, c_time = st.columns([1.5, 1.2, 1])
with c_mode:
    time_mode = st.selectbox("Mode", ["Live (Latest 4h)", "Historical Lookback"])

selected_end_epoch = None
if time_mode == "Historical Lookback":
    with c_date:
        target_date = st.date_input("Date", value=(datetime.now(timezone.utc) - timedelta(days=2)).date(), label_visibility="collapsed")
    with c_time:
        target_time = st.time_input("Time", value=time(12, 0), label_visibility="collapsed")
    dt_combined = datetime.combine(target_date, target_time).replace(tzinfo=timezone.utc)
    selected_end_epoch = int(dt_combined.timestamp())

# Brooks Street Window Sidebar
with st.sidebar:
    st.header("Target Angle Window")
    dir_min = st.number_input("Window Min (° True)", 0, 360, 160)
    dir_max = st.number_input("Window Max (° True)", 0, 360, 230)

@st.cache_data(ttl=900)
def fetch_and_analyze(station, end_epoch):
    hours = 4.0
    rt_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_rt.nc"
    xy_url = f"http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{station}p1_xy.nc"
    
    try:
        ds_rt = xr.open_dataset(rt_url, decode_times=False)
        ds_xy = xr.open_dataset(xy_url, decode_times=False)
    except Exception as e:
        return None, f"CDIP connection failed: {e}"

    if "xyzZDisplacement" not in ds_xy or len(ds_xy.xyzZDisplacement) == 0:
        return None, "Displacement stream currently offline."

    # 1. Match Time Index for Official Spectral Parameters
    rt_times = ds_rt.waveTime.values
    if end_epoch is None:
        t_idx = -1
    else:
        t_idx = int(np.argmin(np.abs(rt_times - end_epoch)))

    try:
        cdip_tp = float(ds_rt.waveTp[t_idx].values)
        cdip_dp = float(ds_rt.waveDp[t_idx].values)
        cdip_hs = float(ds_rt.waveHs[t_idx].values) * 3.28084
        
        # Identify Secondary Wave via Energy Spectrum Peaks
        freqs = ds_rt.waveFrequency.values
        energy = ds_rt.waveEnergyDensity[t_idx].values
        dirs = ds_rt.waveMeanDirection[t_idx].values
        
        peaks, _ = find_peaks(energy, distance=3)
        f_dom = 1.0 / cdip_tp
        f_sec, d_sec = 0.16, 280.0 # defaults
        
        if len(peaks) > 0:
            sorted_p = peaks[np.argsort(energy[peaks])[::-1]]
            for p in sorted_p:
                if abs(freqs[p] - f_dom) > 0.03: # Distinct from dominant
                    f_sec = freqs[p]
                    d_sec = float(dirs[p])
                    break
        t_sec = 1.0 / f_sec
    except Exception:
        cdip_tp, cdip_dp, cdip_hs = 14.0, 180.0, 2.0
        f_dom, f_sec, t_sec, d_sec = 1.0/14.0, 0.16, 6.2, 280.0

    # 2. Slice Displacement Data
    fs = float(ds_xy.xyzSampleRate.values) if "xyzSampleRate" in ds_xy else 1.28
    stride = 2
    eff_fs = fs / stride
    samples_needed = int(hours * 3600 * fs)
    total_len = len(ds_xy.xyzZDisplacement)

    start_time_base = float(ds_xy.xyzStartTime.values) if "xyzStartTime" in ds_xy else datetime.now(timezone.utc).timestamp() - (total_len/fs)

    if end_epoch is None:
        idx_end = total_len
    else:
        idx_end = min(total_len, max(samples_needed, int((end_epoch - start_time_base) * fs)))
    idx_start = max(0, idx_end - samples_needed)

    try:
        z_raw = ds_xy.xyzZDisplacement[idx_start:idx_end:stride].values.astype(np.float64)
    except Exception as e:
        return None, f"Slicing error: {e}"

    z_raw[(z_raw < -900)] = 0.0
    time_min = np.arange(len(z_raw)) / (eff_fs * 60.0)

    # 3. Wave Envelope Extraction
    def extract_sets(f_center, spec_dir, bw=0.025):
        low, high = max(0.035, f_center - bw), min((eff_fs / 2.0) * 0.95, f_center + bw)
        b, a = butter(3, [low, high], btype="band", fs=eff_fs)
        z_f = filtfilt(b, a, z_raw)
        
        env = np.abs(hilbert(z_f))
        env_sm = np.convolve(env, np.hanning(int(eff_fs * 8.0))/np.sum(np.hanning(int(eff_fs * 8.0))), mode="same")

        thresh = float(np.mean(env) + 0.7 * np.std(env))
        p_idx, _ = find_peaks(env, height=thresh, distance=int((1.0 / f_center) * 3.5 * eff_fs))

        pkts = []
        for p in p_idx:
            w = int((1.0 / f_center) * 1.5 * eff_fs)
            s_i, e_i = max(0, p - w), min(len(z_raw), p + w)
            h = (np.max(z_f[s_i:e_i]) - np.min(z_f[s_i:e_i])) * 3.28084
            waves = max(len(find_peaks(z_f[s_i:e_i], distance=int(eff_fs * (1.0 / f_center) * 0.7))[0]), 1)
            valid = (dir_min <= spec_dir <= dir_max) if dir_min <= dir_max else (spec_dir >= dir_min or spec_dir <= dir_max)
            pkts.append({"time_min": time_min[p], "height_ft": h, "waves": waves, "valid": valid})
        
        return z_f * 3.28084, env_sm * 3.28084, thresh * 3.28084, pkts

    z_dom_f, env_dom, thresh_dom, pkts_dom = extract_sets(f_dom, cdip_dp, bw=0.02)
    _, _, _, pkts_sec = extract_sets(f_sec, d_sec, bw=0.035)

    return {
        "time_min": time_min, "cdip_hs": cdip_hs, "cdip_tp": cdip_tp, "cdip_dp": cdip_dp,
        "t_sec": t_sec, "d_sec": d_sec,
        "z_dom_f": z_dom_f, "env_dom": env_dom, "thresh_dom": thresh_dom,
        "pkts_dom": pkts_dom, "pkts_sec": pkts_sec
    }, None

with st.spinner("Processing Spectral Data..."):
    data, err = fetch_and_analyze(station_id, selected_end_epoch)

if err:
    st.error(err)
else:
    # --- Metrics ---
    p_dom = [p for p in data["pkts_dom"] if p["valid"]]
    avg_lull = np.mean([p_dom[i+1]["time_min"] - p_dom[i]["time_min"] for i in range(len(p_dom)-1)]) if len(p_dom) > 1 else 0.0
    avg_dom_h = np.mean([p["height_ft"] for p in p_dom]) if p_dom else 0.0

    p_sec = data["pkts_sec"]
    avg_sec_h = np.mean([p["height_ft"] for p in p_sec]) if p_sec else 0.0

    # Row 1: Bulk Spectral 
    st.markdown("### 📊 CDIP Bulk Spectral Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hs", f"{data['cdip_hs']:.1f} ft")
    c2.metric("Peak Tp", f"{data['cdip_tp']:.0f} s")
    c3.metric("Peak Dp", f"{data['cdip_dp']:.0f}°")

    # Row 2: Band Isolated Dominant
    st.markdown(f"### 🎯 Dominant Band Sets ({data['cdip_tp']:.0f}s / {data['cdip_dp']:.0f}°)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Avg Set Ht", f"{avg_dom_h:.1f} ft" if avg_dom_h > 0 else "—")
    d2.metric("Avg Lull", f"{avg_lull:.1f} min" if avg_lull > 0 else "—")
    d3.metric("Max Wait", f"{np.max([p_dom[i+1]['time_min'] - p_dom[i]['time_min'] for i in range(len(p_dom)-1)]):.0f} m" if len(p_dom) > 1 else "—")
    d4.metric("Set Count", f"{len(p_dom)}")

    # Row 3: Band Isolated Secondary
    st.markdown(f"### 💨 Secondary Wave Band ({data['t_sec']:.0f}s / {data['d_sec']:.0f}°)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Pulse Ht", f"{avg_sec_h:.1f} ft" if avg_sec_h > 0 else "—")
    s2.metric("Spacing", f"{(240.0 / len(p_sec)):.1f} min" if len(p_sec) > 1 else "Continuous")
    s3.metric("Wave Type", "Swell" if data["t_sec"] >= 10.0 else "Windchop")
    s4.metric("Pulse Count", f"{len(p_sec)}")

    # Time-Series Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["time_min"], y=data["z_dom_f"], mode="lines", name=f"{data['cdip_tp']:.0f}s Heave", line=dict(color="rgba(41, 182, 246, 0.35)", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=data["time_min"], y=data["env_dom"], mode="lines", name="Envelope", line=dict(color="#ff9800", width=1.5)))
    fig.add_hline(y=data["thresh_dom"], line=dict(color="#ef5350", dash="dot", width=1))
    if p_dom:
        fig.add_trace(go.Scatter(
            x=[p["time_min"] for p in p_dom], y=[p["height_ft"]/2.0 for p in p_dom],
            mode="markers", name="Set", marker=dict(color="#00e676", size=7, symbol="diamond")
        ))
    fig.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=10)), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Southern California Shadow Map
    st.markdown(f"### 🗺️ Island Shadow Projection ({data['cdip_dp']:.0f}° True)")
    map_fig = go.Figure()

    # Coastline
    coast_lat = [34.45, 34.41, 34.27, 34.02, 33.74, 33.74, 33.60, 33.535, 33.46, 33.19, 32.93, 32.67, 32.55]
    coast_lon = [-120.47, -119.69, -119.29, -118.80, -118.40, -118.11, -117.88, -117.78, -117.70, -117.38, -117.26, -117.24, -117.13]
    map_fig.add_trace(go.Scatter(x=coast_lon, y=coast_lat, mode="lines", line=dict(color="#78909c", width=1.5), name="Coast", hoverinfo="skip"))

    # Islands
    islands = {
        "Catalina": [(33.48, -118.60), (33.43, -118.50), (33.32, -118.32), (33.30, -118.35), (33.35, -118.52), (33.48, -118.60)],
        "San Clemente": [(33.03, -118.60), (32.95, -118.55), (32.81, -118.36), (32.82, -118.42), (33.00, -118.62), (33.03, -118.60)],
        "San Nicolas": [(33.28, -119.58), (33.25, -119.45), (33.22, -119.48), (33.25, -119.59), (33.28, -119.58)]
    }

    # Shadows (Directed from CDIP Dp)
    sh_rad = np.radians((data["cdip_dp"] + 180.0) % 360.0)
    d_lat, d_lon = 1.15 * np.cos(sh_rad), 1.15 * np.sin(sh_rad)

    for name, pts in islands.items():
        sh_x = [p[1] for p in pts] + [p[1] + d_lon for p in reversed(pts)]
        sh_y = [p[0] for p in pts] + [p[0] + d_lat for p in reversed(pts)]
        map_fig.add_trace(go.Scatter(x=sh_x, y=sh_y, fill="toself", fillcolor="rgba(239, 83, 80, 0.3)", line=dict(color="rgba(239, 83, 80, 0.4)", width=1), name=f"{name} Shadow", hoverinfo="skip"))
        map_fig.add_trace(go.Scatter(x=[p[1] for p in pts], y=[p[0] for p in pts], fill="toself", fillcolor="#455a64", line=dict(color="#90a4ae", width=1.5), name=name, hoverinfo="text", text=name))

    # Points
    map_fig.add_trace(go.Scatter(x=[-117.778], y=[33.535], mode="markers+text", marker=dict(size=8, color="#00e676", symbol="star"), text=["Laguna"], textposition="top right", name="Laguna"))
    map_fig.add_trace(go.Scatter(x=[STATIONS[selected_label]["lon"]], y=[STATIONS[selected_label]["lat"]], mode="markers+text", marker=dict(size=7, color="#29b6f6", symbol="circle"), text=[f"Buoy {station_id}"], textposition="bottom center", name="Buoy"))

    # Vector Arrow
    map_fig.add_trace(go.Scatter(
        x=[-118.8, -118.8 + 0.35 * np.sin(sh_rad)], y=[32.5, 32.5 + 0.35 * np.cos(sh_rad)],
        mode="lines+markers", line=dict(color="#00e676", width=2), marker=dict(size=[0, 8], symbol="triangle-up"), name=f"Swell ({data['cdip_dp']:.0f}°)"
    ))

    map_fig.update_layout(
        template="plotly_dark", height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        xaxis=dict(range=[-120.2, -117.0], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[32.4, 34.6], showgrid=False, scaleanchor="x", scaleratio=1.19, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=10, b=10), showlegend=False
    )
    st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': False})
