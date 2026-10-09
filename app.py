
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
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1120px; }
[data-testid="stHeader"] { background: rgba(247,249,252,.92); }
.hero { padding: 28px 30px; border-radius: 18px; background: linear-gradient(120deg,#102f46,#176b68); color: white; margin-bottom: 22px; }
.hero h1 { color: white; margin: 0 0 6px 0; letter-spacing: .04em; font-size: 2.15rem; }
.hero p { color: #e0f2f5; margin: 3px 0 0 0; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #dfe7ef; padding: 17px 18px; border-radius: 14px; box-shadow: 0 2px 8px rgba(16,47,70,.04); }
div[data-testid="stMetricLabel"] { color: #526477; }
div[data-testid="stMetricValue"] { color: #17354d; }
div.stButton > button[kind="primary"] { background: #1769aa; border: 1px solid #1769aa; color: white; border-radius: 10px; min-height: 2.8rem; font-weight: 650; }
div.stButton > button[kind="primary"]:hover { background: #10558c; border-color: #10558c; color: white; }
.result-panel { padding: 18px 20px; border-radius: 14px; border: 1px solid #dfe7ef; background: #fff; min-height: 125px; }
.result-eyebrow { color: #5a6c7f; font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 7px; }
.result-heading { color: #17354d; font-size: 1.25rem; font-weight: 700; margin-bottom: 7px; }
.result-copy { color: #526477; font-size: .94rem; line-height: 1.5; }
.section-intro { color: #607286; margin-top: -.4rem; margin-bottom: 1rem; }
hr { border-color: #e1e8ef; }
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
    st.success(f"**{action_title}** — {action_text}")
elif result["output"] < 50:
    st.info(f"**{action_title}** — {action_text}")
elif result["output"] < 75:
    st.warning(f"**{action_title}** — {action_text}")
else:
    st.error(f"**{action_title}** — {action_text}")

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
