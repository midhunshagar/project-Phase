# ECONOSIM-AI

AI-driven counterfactual economic simulation and policy impact analysis framework.

## What is implemented

- Economic state representation
- Policy scenarios
- Multi-year deterministic simulation
- Synthetic economic dataset generation
- Data loading and cleaning
- GDP machine-learning model selection
- Time-ordered train/test evaluation
- Counterfactual vs baseline comparison
- GDP prediction
- Permutation-based model explainability
- FastAPI backend
- Streamlit dashboard
- Automated tests
- Configurable project structure

## Important scientific limitation

The included dataset is synthetic. The equations are a transparent educational simulation model and are not calibrated to a specific country's real economy. For the final research evaluation, replace/calibrate the dataset and equations using authoritative external data and document the validation procedure.

## Run

Activate the virtual environment, install requirements, then:

```powershell
py -m pip install -r requirements.txt
py run.py
py -m pytest
py -m streamlit run frontend/app.py
```

API:

```powershell
py -m uvicorn app.api.main:app --reload
```

Then open the FastAPI docs at `http://127.0.0.1:8000/docs`.
