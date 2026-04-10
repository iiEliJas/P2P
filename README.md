# P2P
# Supply Chain Analytics: Predictive-to-Prescriptive Framework

A complete machine learning and operations research pipeline for demand forecasting and shipping optimization.

## Overview

This project combines deep learning forecasting with integer linear programming to optimize shipping allocation in supply chain management. The pipeline progresses from exploratory analysis to demand prediction to optimal resource allocation.

## References
This framework implements the research from:

* **Khai Banh Nghiep, Duc Nguyen Minh, Lan Hoang Thi (2026).** *Bridging Deep Learning and Integer Linear Programming: A Predictive-toPrescriptive Framework for Supply Chain Analytics*. 	arXiv:2604.01775 [cs.LG].
* **Link:** [https://doi.org/10.48550/arXiv.2604.01775](https://doi.org/10.48550/arXiv.2604.01775)

## Key Results

// not finished

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/iiEliJas/P2P.git
cd P2P
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download the DataCo Supply Chain dataset
- Place the CSV file in `data/raw/` as `DataCoSupplyChainDataset.csv`
- Update file path in `src/config.py` if needed

### 5. Configuration
Edit `src/config.py` to adjust:
- Data paths
- Train/test split ratios
- Forecasting parameters
- Optimization constraints

## Usage

### Phase 1: EDA
```bash
python run/run_phase1.py
```
Generates visualizations and summary statistics in `results/visualizations/`

### Phase 2: Forecasting
```bash
python run/run_phase2.py
```
Trains MSTL, Baseline and N-BEATS models. Compares performance and selects best model

### Phase 3: ILP
```bash
python run/run_phase3.py
```
Builds ILP model, generates optimal allocation, and compares with baseline strategies

### Notebooks
There are also 3 notebooks demonstrating all phases in detail and step by step.
```bash
- notebooks/01_eda_exploration.ipynb
- notebooks/02_forecasting_comparison.ipynb
- notebooks/03_optimization_analysis.ipynb
```

## Results

Results are automatically generated in `results/`:

### Visualizations
Visualizations will be placed here for showcase.

### Forecasts
Forecast outputs saved in `results/forecasts/`


## Testing

```bash
# run all tests
pytest

# run specific tes
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Dependencies

- **Data**: pandas, numpy, scipy
- **Statistical**: statsmodels, scikit-learn
- **ML**: torch, pytorch-lightning, pytorch-forecasting
- **Optimization**: PuLP
- **Visualization**: plotly, matplotlib, seaborn
- **Testing**: pytest

See `requirements.txt` for versions


## Future Plans

- Stochastic optimization (robust ILP)
- Multiple product SKUs
- Exogenous variables (holidays, promotions)
- Real-time deployment

## License

MIT License
