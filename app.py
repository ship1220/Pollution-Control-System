
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 16px;
        border-radius: 12px;
    }
    .hero {
        padding: 25px;
        border-radius: 16px;
        background: linear-gradient(120deg, #123b36, #176b55);
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { color: white; margin-bottom: 5px; }
    .hero p { color: #e0f2e9; margin-bottom: 0; }
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

st.sidebar.info(
    "AQI is fetched for the selected location. The industrial "
    "emission percentage is supplied separately."
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

    st.markdown("### How the system works")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Fuzzification**\n\nConvert AQI and emission inputs into fuzzy membership degrees.")
    c2.markdown("**2. Mamdani inference**\n\nEvaluate 18 fuzzy rules and aggregate the output sets.")
    c3.markdown("**3. Defuzzification**\n\nCalculate a crisp control intensity using the centroid method.")

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
m3.metric("Fuzzy control intensity", f"{result['output']:.2f}%")

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
# FUZZIFICATION AND MEMBERSHIP FUNCTION PLOTS
# ============================================================

st.divider()
st.subheader("Fuzzification & Membership Functions")

st.write(
    "The input values may belong partially to multiple fuzzy sets. "
    "The membership degree ranges from 0 to 1."
)

model = build_fuzzy_system()
p1, p2 = st.columns(2)

with p1:
    fig, ax = plt.subplots(figsize=(7, 4))
    model["aqi"].view(ax=ax)
    ax.axvline(
        result["aqi"], color="black", linestyle="--",
        label=f"Input = {result['aqi']:.1f}"
    )
    ax.set_title("Ambient US AQI")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with p2:
    fig, ax = plt.subplots(figsize=(7, 4))
    model["emissions"].view(ax=ax)
    ax.axvline(
        result["emission"], color="black", linestyle="--",
        label=f"Input = {result['emission']:.0f}%"
    )
    ax.set_title("Industrial emission level")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

membership_rows = [
    {"Input": "AQI", "Fuzzy set": k, "Membership degree": v}
    for k, v in result["aqi_mu"].items()
] + [
    {"Input": "Emissions", "Fuzzy set": k, "Membership degree": v}
    for k, v in result["emission_mu"].items()
]

st.dataframe(
    pd.DataFrame(membership_rows).round(4),
    hide_index=True,
    use_container_width=True,
)

# ============================================================
# RULE AUDIT
# ============================================================

st.divider()
st.subheader("Fuzzy Rule Evaluation")

st.write(
    f"{len(result['active_rules'])} of {model['rule_count']} rules "
    "have non-zero firing strength for the current inputs."
)

st.dataframe(
    result["active_rules"].round(4),
    hide_index=True,
    use_container_width=True,
)

with st.expander("View complete rule base"):
    st.dataframe(
        result["rule_audit"].round(4),
        hide_index=True,
        use_container_width=True,
    )

# ============================================================
# OUTPUT MEMBERSHIP FUNCTIONS
# ============================================================

st.divider()
st.subheader("Defuzzification Result")

fig, ax = plt.subplots(figsize=(10, 4))
model["control"].view(ax=ax)
ax.axvline(
    result["output"], color="red", linestyle="--",
    label=f"Centroid output = {result['output']:.2f}%"
)
ax.set_title("Pollution-control output membership functions")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.progress(
    min(max(int(round(result["output"])), 0), 100),
    text=f"Recommended control intensity: {result['output']:.2f}%",
)

# ============================================================
# ENGINEERING NOTES
# ============================================================

with st.expander("Model assumptions and limitations"):
    st.markdown("""
    - **Ambient AQI:** retrieved for the selected geographic location;
      it does not directly measure emissions from an individual factory.
    - **AQI scale:** the model uses US AQI and its approximate categories.
    - **Industrial input:** emission level is a user-supplied percentage
      of a defined reference level, not a pollutant concentration.
    - **Fuzzy rules:** the 18 rules and membership functions are initial
      academic design parameters and require expert validation.
    - **Control output:** the percentage is a decision-support score,
      not a direct equipment setpoint.
    - **Data source:** the air-quality API can return model-derived
      current estimates rather than direct monitoring-station readings.
    """)

st.caption(
    "AIRGUARD · Academic fuzzy-logic prototype · "
    "Mamdani inference with centroid defuzzification"
)
