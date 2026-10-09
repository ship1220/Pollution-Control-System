import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Pollution Control System — Mamdani fuzzy inference prototype
# Uses NumPy for fuzzy inference; no scikit-fuzzy or SciPy.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Pollution Control System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Neutral engineering-dashboard styling. Red is reserved for critical alerts.
st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      [data-testid="stHeader"] { background: rgba(245,247,251,0.92); }
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1200px; }
      h1, h2, h3 { color: #15263d; }
      .hero {
        padding: 1.25rem 1.4rem; border-radius: 16px;
        background: linear-gradient(120deg, #e9f2ff, #f7fbff);
        border: 1px solid #d5e5fb; margin-bottom: 1rem;
      }
      .hero p { color: #4a5d73; margin-bottom: 0; }
      div.stButton > button[kind="primary"] {
        background: #1769aa; border: 1px solid #1769aa; color: white;
        border-radius: 9px; font-weight: 600; min-height: 2.7rem;
      }
      div.stButton > button[kind="primary"]:hover {
        background: #10558c; border-color: #10558c; color: white;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e0e7ef;
        padding: 0.85rem 1rem; border-radius: 12px;
      }
      .small-note { color: #617187; font-size: 0.88rem; }
      .result-card {
        background: white; border: 1px solid #e0e7ef; border-radius: 12px;
        padding: 1rem 1.1rem; height: 100%;
      }
      .result-label { color: #617187; font-size: 0.85rem; margin-bottom: 0.3rem; }
      .result-value { color: #15263d; font-size: 1.65rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Fuzzy membership functions ----------

def trimf(x, abc):
    """Triangular membership function; works with scalar or NumPy array."""
    a, b, c = [float(v) for v in abc]
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    if a != b:
        mask = (a < x) & (x <= b)
        y[mask] = (x[mask] - a) / (b - a)
    else:
        y[x == a] = 1.0

    if b != c:
        mask = (b < x) & (x < c)
        y[mask] = (c - x[mask]) / (c - b)
    else:
        y[x == b] = 1.0

    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x, abcd):
    """Trapezoidal membership function; works with scalar or NumPy array."""
    a, b, c, d = [float(v) for v in abcd]
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    if a != b:
        mask = (a < x) & (x < b)
        y[mask] = (x[mask] - a) / (b - a)
    else:
        y[x == a] = 1.0

    y[(b <= x) & (x <= c)] = 1.0

    if c != d:
        mask = (c < x) & (x < d)
        y[mask] = (d - x[mask]) / (d - c)
    else:
        y[x == d] = 1.0

    return np.clip(y, 0.0, 1.0)


def membership(value, definition):
    kind, params = definition
    fn = trimf if kind == "tri" else trapmf
    return float(fn(np.array([value], dtype=float), params)[0])


AQI_SETS = {
    "Good": ("trap", (0, 0, 25, 50)),
    "Moderate": ("trap", (30, 50, 100, 130)),
    "Poor": ("trap", (80, 120, 180, 220)),
    "Severe": ("trap", (170, 230, 500, 500)),
}
EMISSION_SETS = {
    "Low": ("trap", (0, 0, 25, 45)),
    "Moderate": ("trap", (30, 45, 60, 75)),
    "High": ("trap", (60, 75, 85, 95)),
    "Very high": ("trap", (85, 95, 100, 100)),
}
CONTROL_SETS = {
    "Low": ("trap", (0, 0, 18, 38)),
    "Moderate": ("trap", (25, 40, 52, 68)),
    "High": ("trap", (55, 68, 78, 90)),
    "Very high": ("trap", (80, 92, 100, 100)),
}

# Rows: ambient AQI; columns: estimated industrial emission level.
# This is an explicitly heuristic, illustrative rule base and must be tuned
# against plant-specific measurements before any real-world use.
RULE_MATRIX = {
    "Good":      {"Low": "Low", "Moderate": "Low", "High": "Moderate", "Very high": "High"},
    "Moderate":  {"Low": "Low", "Moderate": "Moderate", "High": "High", "Very high": "Very high"},
    "Poor":      {"Low": "Moderate", "Moderate": "High", "High": "High", "Very high": "Very high"},
    "Severe":    {"Low": "High", "Moderate": "High", "High": "Very high", "Very high": "Very high"},
}

AQI_NAMES = list(AQI_SETS)
EMISSION_NAMES = list(EMISSION_SETS)
CONTROL_NAMES = list(CONTROL_SETS)


def run_fuzzy_inference(aqi, emission):
    """Mamdani min-AND, max aggregation, centroid defuzzification."""
    aqi_mu = {name: membership(aqi, definition) for name, definition in AQI_SETS.items()}
    emission_mu = {
        name: membership(emission, definition)
        for name, definition in EMISSION_SETS.items()
    }

    output_x = np.linspace(0, 100, 1001)
    aggregated = np.zeros_like(output_x)
    fired_rules = []

    for aqi_name in AQI_NAMES:
        for emission_name in EMISSION_NAMES:
            strength = min(aqi_mu[aqi_name], emission_mu[emission_name])
            if strength <= 0:
                continue
            consequent = RULE_MATRIX[aqi_name][emission_name]
            output_definition = CONTROL_SETS[consequent]
            output_curve = trapmf(output_x, output_definition[1])
            clipped = np.minimum(strength, output_curve)
            aggregated = np.maximum(aggregated, clipped)
            fired_rules.append({
                "Ambient AQI": aqi_name,
                "Emissions": emission_name,
                "Rule output": consequent,
                "Firing strength": strength,
            })

    denominator = float(np.sum(aggregated))
    if denominator <= 0:
        control = 0.0
    else:
        control = float(np.sum(output_x * aggregated) / denominator)

    if control < 30:
        label = "Low"
    elif control < 56:
        label = "Moderate"
    elif control < 79:
        label = "High"
    else:
        label = "Very high"

    return {
        "control": control,
        "label": label,
        "aqi_memberships": aqi_mu,
        "emission_memberships": emission_mu,
        "output_x": output_x,
        "aggregated": aggregated,
        "fired_rules": sorted(fired_rules, key=lambda row: row["Firing strength"], reverse=True),
    }


def get_location(query):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(
        url,
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        timeout=12,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        raise ValueError(f"No location found for “{query}”. Try a nearby city name.")
    return results[0]


def get_live_aqi(latitude, longitude):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    response = requests.get(
        url,
        params={"latitude": latitude, "longitude": longitude, "current": "us_aqi"},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    current = data.get("current") or {}
    aqi = current.get("us_aqi")
    if aqi is None:
        raise ValueError("The air-quality service did not return a current US AQI value.")
    return float(aqi), current.get("time", "Current observation")


def plot_memberships(sets, x_min, x_max, title, x_label):
    x = np.linspace(x_min, x_max, 501)
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    for name, (kind, params) in sets.items():
        fn = trimf if kind == "tri" else trapmf
        ax.plot(x, fn(x, params), linewidth=2, label=name)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Membership")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.22)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig


def plot_aggregated_output(result):
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.fill_between(result["output_x"], result["aggregated"], alpha=0.22)
    ax.plot(result["output_x"], result["aggregated"], linewidth=2)
    ax.axvline(result["control"], linestyle="--", linewidth=1.5, label=f'Centroid: {result["control"]:.0f}%')
    ax.set_title("Aggregated output membership", loc="left", fontsize=11, pad=10)
    ax.set_xlabel("Recommended control intensity (%)")
    ax.set_ylabel("Membership")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------- Page content ----------

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0 0 .35rem 0;">🌿 Pollution Control System</h1>
      <p>Mamdani fuzzy inference for an illustrative industrial emissions-control recommendation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System inputs")
    input_mode = st.radio(
        "Ambient air-quality data",
        ["Fetch live AQI by location", "Enter AQI manually"],
        help="Live AQI is an ambient indicator and is not a direct measurement of the plant's stack emissions.",
    )

    location_name = ""
    observation_time = "Manual input"
    if input_mode == "Fetch live AQI by location":
        location_name = st.text_input("City or location", value="Mumbai")
        st.caption("Uses Open-Meteo geocoding and air-quality services.")
    else:
        manual_aqi = st.slider("Ambient US AQI", min_value=0, max_value=300, value=95, step=1)

    emission = st.slider(
        "Industrial emission level (%)",
        min_value=0,
        max_value=100,
        value=39,
        step=1,
        help="Illustrative percentage relative to a reference emission level defined for this prototype; not a measured regulatory emission concentration.",
    )

    analyze = st.button("Analyze & Recommend", type="primary", use_container_width=True)

st.caption("Decision-support prototype · Mamdani inference · Min AND · Max aggregation · Centroid defuzzification")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "last_location" not in st.session_state:
    st.session_state.last_location = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = ""

if analyze:
    try:
        if input_mode == "Fetch live AQI by location":
            if not location_name.strip():
                st.error("Enter a city or location first.")
                st.stop()
            location = get_location(location_name.strip())
            aqi, observation_time = get_live_aqi(location["latitude"], location["longitude"])
            resolved_location = ", ".join(
                part for part in [location.get("name"), location.get("admin1"), location.get("country")] if part
            )
        else:
            aqi = float(manual_aqi)
            resolved_location = "Manual AQI input"

        result = run_fuzzy_inference(aqi, emission)
        st.session_state.last_result = result
        st.session_state.last_inputs = {"aqi": aqi, "emission": emission}
        st.session_state.last_location = resolved_location
        st.session_state.last_time = observation_time
    except requests.RequestException:
        st.error("Could not reach the live AQI service. Try again or switch to manual AQI input.")
    except ValueError as exc:
        st.error(str(exc))

if st.session_state.last_result is None:
    st.info("Choose the input values in the sidebar, then select **Analyze & Recommend**.")
    st.subheader("How the controller reasons")
    st.markdown(
        "- Fuzzifies the ambient AQI and emission-level inputs.\n"
        "- Evaluates the fuzzy rule base using minimum firing strength.\n"
        "- Aggregates consequent sets with maximum and defuzzifies using the centroid."
    )
else:
    result = st.session_state.last_result
    inputs = st.session_state.last_inputs
    control = result["control"]
    # Rounded values are for presentation; centroid retains its full precision internally.
    control_display = int(round(control))
    pump_rpm = int(round(600 + (control / 100.0) * 1200))
    esp_kv = int(round(20 + (control / 100.0) * 40))

    st.subheader("Recommendation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Ambient US AQI", f'{inputs["aqi"]:.0f}', st.session_state.last_location)
    m2.metric("Industrial emission level", f'{inputs["emission"]:.0f}%')
    m3.metric("Recommended control intensity", f"{control_display}%", result["label"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Simulated scrubber pump speed</div>
              <div class="result-value">{pump_rpm:,} RPM</div>
              <div class="small-note">Illustrative mapping: 600–1,800 RPM across 0–100% control.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Simulated electrostatic precipitator voltage</div>
              <div class="result-value">{esp_kv} kV</div>
              <div class="small-note">Illustrative mapping: 20–60 kV across 0–100% control.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result["label"] == "Very high":
        st.error("Very high simulated control recommendation. Review plant-specific measurements and safety procedures.")
    elif result["label"] == "High":
        st.warning("High simulated control recommendation. Verify with validated plant instrumentation and operating limits.")
    else:
        st.success("No critical simulated-control state was triggered by this rule base.")

    st.caption(
        f"Observation: {st.session_state.last_time}. "
        "Actuator values are illustrative simulations, not real equipment setpoints."
    )

    st.subheader("Fuzzy inference details")
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_memberships(AQI_SETS, 0, 300, "Ambient AQI membership functions", "US AQI"), use_container_width=True)
        st.pyplot(plot_memberships(EMISSION_SETS, 0, 100, "Emission-level membership functions", "Emission level (%)"), use_container_width=True)
    with right:
        st.pyplot(plot_memberships(CONTROL_SETS, 0, 100, "Control-intensity output sets", "Control intensity (%)"), use_container_width=True)
        st.pyplot(plot_aggregated_output(result), use_container_width=True)

    st.subheader("Input membership degrees")
    d1, d2 = st.columns(2)
    with d1:
        st.dataframe(
            pd.DataFrame([{"AQI set": name, "Membership": round(value, 3)} for name, value in result["aqi_memberships"].items()]),
            hide_index=True, use_container_width=True,
        )
    with d2:
        st.dataframe(
            pd.DataFrame([{"Emission set": name, "Membership": round(value, 3)} for name, value in result["emission_memberships"].items()]),
            hide_index=True, use_container_width=True,
        )

    st.subheader("Activated rules")
    if result["fired_rules"]:
        rules_df = pd.DataFrame(result["fired_rules"])
        rules_df["Firing strength"] = rules_df["Firing strength"].round(3)
        st.dataframe(rules_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No rules fired for these inputs. Review the membership-function coverage.")

    with st.expander("View complete fuzzy rule base"):
        rows = []
        for aqi_name in AQI_NAMES:
            row = {"Ambient AQI": aqi_name}
            row.update(RULE_MATRIX[aqi_name])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Method and limitations"):
        st.markdown(
            """
            - **Inference:** Mamdani min-AND, max aggregation, centroid defuzzification.
            - **Inputs:** ambient US AQI and a user-supplied, normalized emission-level percentage.
            - **Output:** a heuristic control-intensity recommendation mapped to illustrative actuator variables.
            - **Not for live control:** AQI is not a plant stack sensor. Membership functions, rules, RPM and voltage ranges must be validated against plant-specific engineering data, emissions limits, equipment curves, and safety interlocks before operational use.
            """
        )

st.divider()
st.caption("Prototype only · Do not connect this recommendation directly to industrial actuators.")
