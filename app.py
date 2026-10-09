
import requests
import numpy as np
import pandas as pd
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIRGUARD | Pollution Control",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --ag-accent: #167d8d;
        --ag-accent-hover: #116775;
        --ag-ink: #172b3a;
        --ag-muted: #5c6d7e;
        --ag-border: #d9e3eb;
        --ag-card-light: #ffffff;
        --ag-card-dark: #202a35;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1160px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        padding: 27px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #12354d 0%, #176b68 100%);
        color: #ffffff;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 24px rgba(10, 35, 50, 0.12);
    }
    .hero h1 {
        color: #ffffff !important;
        margin: 0 0 6px 0;
        letter-spacing: .035em;
        font-size: 2.1rem;
    }
    .hero p {
        color: #e3f2f4 !important;
        margin: 3px 0 0 0;
    }

    /* Streamlit metric cards: explicit contrast in light and dark mode. */
    [data-testid="stMetric"] {
        background: var(--ag-card-light);
        border: 1px solid var(--ag-border);
        border-radius: 14px;
        padding: 17px 18px;
        box-shadow: 0 3px 10px rgba(18, 53, 77, 0.045);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: var(--ag-muted) !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: var(--ag-ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--ag-muted) !important;
    }

    /* Neutral teal accent for the slider, not emergency red. */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: var(--ag-accent) !important;
        border-color: var(--ag-accent) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: var(--ag-accent) !important;
    }

    div.stButton > button[kind="primary"] {
        background: #1769aa;
        border: 1px solid #1769aa;
        color: #ffffff;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 650;
        transition: background .15s ease, border-color .15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #10558c;
        border-color: #10558c;
        color: #ffffff;
    }

    .result-panel {
        padding: 19px 20px;
        border-radius: 14px;
        border: 1px solid var(--ag-border);
        background: var(--ag-card-light);
        min-height: 128px;
    }
    .result-eyebrow {
        color: var(--ag-muted) !important;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .07em;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-heading {
        color: var(--ag-ink) !important;
        font-size: 1.22rem;
        font-weight: 750;
        margin-bottom: 7px;
    }
    .result-copy {
        color: var(--ag-muted) !important;
        font-size: .94rem;
        line-height: 1.55;
    }
    .section-intro {
        color: var(--ag-muted);
        margin-top: -.4rem;
        margin-bottom: 1rem;
    }

    /* Replace harsh alert colors with calm, readable status treatments. */
    .status-panel {
        border-radius: 12px;
        padding: 15px 18px;
        margin-top: 18px;
        border: 1px solid #c9dce7;
        background: #edf5f8;
        color: #1d455b;
        line-height: 1.55;
    }
    .status-panel strong { color: #173b50; }
    .status-panel.low {
        background: #edf7f2;
        border-color: #c9e6d7;
        color: #245b43;
    }
    .status-panel.medium {
        background: #edf5f8;
        border-color: #c9dce7;
        color: #1d455b;
    }
    .status-panel.high {
        background: #fff5e8;
        border-color: #f0d7b2;
        color: #744a16;
    }
    .status-panel.critical {
        background: #fbeceb;
        border-color: #edc4c1;
        color: #8a2925;
    }
    .status-panel.low strong { color: #245b43; }
    .status-panel.medium strong { color: #1d455b; }
    .status-panel.high strong { color: #744a16; }
    .status-panel.critical strong { color: #8a2925; }

    hr { border-color: var(--ag-border); }

    /* Dark theme overrides so cards retain strong text contrast. */
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"], .result-panel {
            background: #202a35;
            border-color: #3b4a58;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        .result-eyebrow, .result-copy {
            color: #c4d0da !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div,
        .result-heading {
            color: #f3f7fa !important;
        }
        .status-panel.low { background: #19372c; border-color: #315b48; color: #d1eadb; }
        .status-panel.medium { background: #1c3544; border-color: #35576b; color: #d4e8f1; }
        .status-panel.high { background: #45351f; border-color: #6b512e; color: #f4dfbd; }
        .status-panel.critical { background: #452725; border-color: #713a37; color: #f6d5d2; }
        .status-panel strong { color: inherit !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LIVE DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def geocode_location(query):
    """Find locations using the Open-Meteo geocoding API."""
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": query,
            "count": 10,
            "language": "en",
            "format": "json",
        },
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        raise ValueError("No matching location was found.")

    return results


@st.cache_data(ttl=900, show_spinner=False)
def fetch_air_quality(latitude, longitude):
    """Fetch the latest available ambient air-quality data."""
    response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "us_aqi,pm2_5,pm10,nitrogen_dioxide,"
                "sulphur_dioxide,ozone"
            ),
            "timezone": "auto",
        },
        timeout=25,
    )
    response.raise_for_status()
    current = response.json().get("current", {})

    value = current.get("us_aqi")
    if value is None:
        raise ValueError("The API returned no current AQI value.")

    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 500:
        raise ValueError("The returned AQI is outside the model range.")

    return current, value


# ============================================================
# MAMDANI FUZZY INFERENCE SYSTEM
# ============================================================

@st.cache_resource
def build_fuzzy_system():
    """Create the reusable fuzzy variables, sets and rules."""

    aqi = ctrl.Antecedent(
        np.arange(0, 501, 1), "aqi"
    )
    emissions = ctrl.Antecedent(
        np.arange(0, 101, 1), "emissions"
    )
    control = ctrl.Consequent(
        np.arange(0, 101, 1), "control"
    )

    # AQI sets: approximate US AQI categories with overlap.
    aqi["good"] = fuzz.trapmf(
        aqi.universe, [0, 0, 25, 55]
    )
    aqi["moderate"] = fuzz.trimf(
        aqi.universe, [40, 75, 110]
    )
    aqi["sensitive"] = fuzz.trimf(
        aqi.universe, [90, 125, 160]
    )
    aqi["unhealthy"] = fuzz.trimf(
        aqi.universe, [140, 175, 210]
    )
    aqi["very_unhealthy"] = fuzz.trimf(
        aqi.universe, [190, 250, 310]
    )
    aqi["hazardous"] = fuzz.trapmf(
        aqi.universe, [280, 360, 500, 500]
    )

    # Emission input: percentage of a defined reference level.
    emissions["low"] = fuzz.trapmf(
        emissions.universe, [0, 0, 15, 35]
    )
    emissions["medium"] = fuzz.trimf(
        emissions.universe, [20, 50, 80]
    )
    emissions["high"] = fuzz.trapmf(
        emissions.universe, [65, 85, 100, 100]
    )

    # Recommended control intensity.
    control["low"] = fuzz.trapmf(
        control.universe, [0, 0, 15, 35]
    )
    control["moderate"] = fuzz.trimf(
        control.universe, [20, 45, 65]
    )
    control["high"] = fuzz.trimf(
        control.universe, [50, 70, 85]
    )
    control["very_high"] = fuzz.trapmf(
        control.universe, [75, 90, 100, 100]
    )

    aqi_terms = [
        "good", "moderate", "sensitive",
        "unhealthy", "very_unhealthy", "hazardous"
    ]
    emission_terms = ["low", "medium", "high"]

    # Rows = AQI terms; columns = emission terms.
    rule_matrix = [
        ["low",       "moderate",  "high"],
        ["moderate",  "high",      "very_high"],
        ["high",      "high",      "very_high"],
        ["high",      "very_high", "very_high"],
        ["very_high", "very_high", "very_high"],
        ["very_high", "very_high", "very_high"],
    ]

    rules = []
    for i, aqi_term in enumerate(aqi_terms):
        for j, emission_term in enumerate(emission_terms):
            rules.append(
                ctrl.Rule(
                    aqi[aqi_term] & emissions[emission_term],
                    control[rule_matrix[i][j]],
                    label=f"R{i * 3 + j + 1:02d}",
                )
            )

    system = ctrl.ControlSystem(rules)

    return {
        "aqi": aqi,
        "emissions": emissions,
        "control": control,
        "system": system,
        "aqi_terms": aqi_terms,
        "emission_terms": emission_terms,
        "rule_matrix": rule_matrix,
        "rule_count": len(rules),
    }


def evaluate_fis(model, aqi_value, emission_value):
    """Run Mamdani inference and return the crisp output and audit."""

    simulation = ctrl.ControlSystemSimulation(model["system"])
    simulation.input["aqi"] = float(aqi_value)
    simulation.input["emissions"] = float(emission_value)
    simulation.compute()

    output = float(simulation.output["control"])

    if not np.isfinite(output):
        raise ValueError("Fuzzy inference returned an invalid output.")

    # Calculate input memberships for explanation and auditing.
    aqi_memberships = {
        term: float(fuzz.interp_membership(
            model["aqi"].universe,
            model["aqi"][term].mf,
            aqi_value,
        ))
        for term in model["aqi_terms"]
    }

    emission_memberships = {
        term: float(fuzz.interp_membership(
            model["emissions"].universe,
            model["emissions"][term].mf,
            emission_value,
        ))
        for term in model["emission_terms"]
    }

    # Mamdani AND: minimum of the antecedent memberships.
    audit = []
    for i, aqi_term in enumerate(model["aqi_terms"]):
        for j, emission_term in enumerate(model["emission_terms"]):
            strength = min(
                aqi_memberships[aqi_term],
                emission_memberships[emission_term],
            )
            audit.append({
                "Rule": f"R{i * 3 + j + 1:02d}",
                "AQI set": aqi_term.replace("_", " ").title(),
                "Emission set": emission_term.title(),
                "Output set": model["rule_matrix"][i][j].replace(
                    "_", " "
                ).title(),
                "Firing strength": strength,
            })

    audit_df = pd.DataFrame(audit)
    active_rules = audit_df[
        audit_df["Firing strength"] > 0
    ].sort_values("Firing strength", ascending=False)

    return (
        output, aqi_memberships, emission_memberships,
        audit_df, active_rules
    )


# ============================================================
# OUTPUT INTERPRETATION
# ============================================================

def describe_action(value):
    """Translate the crisp FIS output into readable guidance."""

    if value < 25:
        return (
            "Low control intensity",
            "Maintain routine monitoring and normal approved operation.",
        )
    elif value < 50:
        return (
            "Moderate control intensity",
            "Review filtration performance and increase monitoring.",
        )
    elif value < 75:
        return (
            "High control intensity",
            "Recommend enhanced filtration and scrubbing within "
            "approved equipment limits.",
        )

    return (
        "Very high control intensity",
        "Recommend maximum approved control operation and investigate "
        "the emission source.",
    )


# ============================================================
# DASHBOARD HEADER AND SIDEBAR
# ============================================================

st.markdown("""
<div class="hero">
    <h1>AIRGUARD</h1>
    <p>Industrial Pollution Control System</p>
    <p>Live air quality · Mamdani fuzzy inference · Control recommendations</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Assessment Inputs")
st.sidebar.caption("Configure the location and industrial input.")

location_query = st.sidebar.text_input(
    "City or location",
    value="Mumbai",
    placeholder="e.g. Mumbai, Virar, Pune",
)

emission_value = st.sidebar.slider(
    "Industrial emission level (%)",
    min_value=0,
    max_value=100,
    value=68,
    help=(
        "Enter the measured or assumed percentage relative to "
        "your defined reference emission level."
    ),
)

st.sidebar.caption(
    "Ambient AQI is fetched for the location. Set the plant emission "
    "level separately using the slider."
)

analyze = st.sidebar.button(
    "Analyze & Recommend",
    type="primary",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.caption("Inference: Mamdani | Defuzzification: Centroid")

# ============================================================
# ANALYSIS
# ============================================================

if analyze:
    if not location_query.strip():
        st.error("Enter a city or location.")
        st.stop()

    try:
        with st.spinner("Finding location and retrieving air quality..."):
            locations = geocode_location(location_query.strip())

            # Choose the first geocoding result.
            # The resolved location is displayed for verification.
            location = locations[0]

            current, aqi_value = fetch_air_quality(
                location["latitude"],
                location["longitude"],
            )

        model = build_fuzzy_system()

        with st.spinner("Running the fuzzy inference engine..."):
            (
                output, aqi_mu, emission_mu,
                rule_audit, active_rules
            ) = evaluate_fis(model, aqi_value, emission_value)

        city = location.get("name", location_query)
        admin = location.get("admin1", "")
        country = location.get("country", "")
        resolved_location = ", ".join(
            item for item in [city, admin, country] if item
        )

        st.session_state["assessment"] = {
            "location": resolved_location,
            "aqi": aqi_value,
            "emission": emission_value,
            "timestamp": current.get("time", "Not available"),
            "current": current,
            "output": output,
            "aqi_mu": aqi_mu,
            "emission_mu": emission_mu,
            "rule_audit": rule_audit,
            "active_rules": active_rules,
        }

    except requests.RequestException:
        st.error(
            "The live-data service could not be reached. "
            "Check your internet connection and try again."
        )
        st.stop()

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Assessment failed: {error}")
        st.stop()


# Keep results visible when a widget changes or the page reruns.
if "assessment" not in st.session_state:
    st.info(
        "Enter a location and emission level in the sidebar, then click "
        "**Analyze & Recommend** to run the fuzzy system."
    )

    st.markdown("### How to use AIRGUARD")
    st.markdown(
        "1. Enter a city or location.\n"
        "2. Set the industrial emission level.\n"
        "3. Select **Analyze & Recommend** to view the expected control response."
    )
    st.caption("The app shows the recommendation and key input readings without technical graphs or rule tables.")
    st.stop()


result = st.session_state["assessment"]

st.caption(
    f"Resolved location: {result['location']} · "
    f"API timestamp: {result['timestamp']}"
)

# ============================================================
# KEY METRICS AND CONTROL ACTION
# ============================================================

st.subheader("Assessment Overview")

m1, m2, m3 = st.columns(3)
m1.metric("Ambient US AQI", f"{result['aqi']:.1f}")
m2.metric("Industrial emission input", f"{result['emission']:.0f}%")
m3.metric('Recommended control intensity', str(round(result['output'])) + '%')

action_title, action_text = describe_action(result["output"])

if result["output"] < 25:
    status_class = "low"
elif result["output"] < 50:
    status_class = "medium"
elif result["output"] < 75:
    status_class = "high"
else:
    status_class = "critical"

st.markdown(
    f"""
    <div class="status-panel {status_class}">
        <strong>{action_title}</strong> — {action_text}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Control intensity is a simulated recommendation, not a validated "
    "physical actuator setting."
)

# ============================================================
# POLLUTANT DATA
# ============================================================

with st.expander("View ambient pollutant readings"):
    pollutant_labels = {
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "nitrogen_dioxide": "Nitrogen dioxide",
        "sulphur_dioxide": "Sulphur dioxide",
        "ozone": "Ozone",
    }

    pollutant_rows = [
        {
            "Pollutant": label,
            "Reported value": result["current"].get(key),
        }
        for key, label in pollutant_labels.items()
        if result["current"].get(key) is not None
    ]

    if pollutant_rows:
        st.dataframe(
            pd.DataFrame(pollutant_rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("No additional pollutant readings were returned.")


# ============================================================
# EXPECTED RESULTS — SIMPLE OPERATOR VIEW
# ============================================================

st.divider()
st.subheader("Expected Results")
st.markdown(
    '<p class="section-intro">A plain-language summary of the recommendation for the selected inputs.</p>',
    unsafe_allow_html=True,
)

result_col1, result_col2 = st.columns(2)
with result_col1:
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-eyebrow">Recommended response</div>
            <div class="result-heading">{action_title}</div>
            <div class="result-copy">{action_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with result_col2:
    if result["output"] < 25:
        response_level = "Routine"
        response_detail = "Continue normal approved operation and routine monitoring."
    elif result["output"] < 50:
        response_level = "Increased monitoring"
        response_detail = "Review filtration performance and monitor the emission trend."
    elif result["output"] < 75:
        response_level = "Enhanced control"
        response_detail = "Review filtration and scrubbing performance within approved equipment limits."
    else:
        response_level = "Priority review"
        response_detail = "Review the emission source and follow the plant's approved high-control procedures."

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-eyebrow">Suggested next step</div>
            <div class="result-heading">{response_level}</div>
            <div class="result-copy">{response_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("#### Input summary")
s1, s2, s3 = st.columns(3)
s1.metric("Selected location", result["location"])
s2.metric("Ambient US AQI", f"{result['aqi']:.0f}")
s3.metric("Industrial emission input", f"{result['emission']:.0f}%")

# ============================================================
# ENGINEERING NOTES
# ============================================================

with st.expander("About this recommendation"):
    st.markdown("""
    - **Ambient AQI** describes air quality around the selected location; it does not directly measure one factory's emissions.
    - **Industrial emission level** is a user-supplied percentage of a reference level defined for this prototype.
    - **Fuzzy logic** combines the two inputs to calculate the displayed control recommendation.
    - **Prototype only:** the rules are illustrative and need validation with plant-specific data. Do not connect the recommendation directly to equipment.
    """)

st.caption(
    "AIRGUARD · Academic fuzzy-logic prototype · "
    "Mamdani inference with centroid defuzzification"
)
