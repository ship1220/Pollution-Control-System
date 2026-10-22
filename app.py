
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
if result["aqi"] >= 151:
    st.warning(
        "Ambient air quality is very unhealthy or worse. Investigate local "
        "air-quality conditions separately; ambient AQI alone does not prove "
        "that this plant caused the pollution."
    )
elif result["aqi"] >= 101:
    st.info(
        "Ambient air quality is unhealthy for sensitive groups or worse. "
        "Treat this as a contextual alert, not a direct measurement of this "
        "plant's emissions."
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
            width="stretch",
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
