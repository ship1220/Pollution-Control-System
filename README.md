# Pollution Control System — Updated Streamlit App

## Files
- `app.py`: Streamlit interface and NumPy-based Mamdani fuzzy inference.
- `requirements.txt`: lightweight dependencies; `scikit-fuzzy` and `scipy` are intentionally removed to avoid their dependency/build issues on Streamlit Community Cloud.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Replace the existing `app.py` and `requirements.txt` in your GitHub repository with these files.
2. Commit and push the changes.
3. In Streamlit Community Cloud, select **Reboot app** or wait for the app to rebuild.
4. Check the logs if deployment fails.

## Fuzzy inference
The app implements Mamdani inference directly:
- membership functions for ambient US AQI, emission level, and control intensity;
- minimum firing strength for AND;
- maximum aggregation across rule consequents;
- centroid defuzzification.

The rule base is an illustrative heuristic and should be tuned with validated plant-specific data. Ambient AQI is not a direct measure of industrial stack emissions. The simulated scrubber RPM and precipitator voltage are display-only illustrative mappings, not safe operational setpoints.
