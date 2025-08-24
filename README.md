# Stock Price Prediction

A comprehensive Python system for predicting stock prices using multiple machine learning models with uncertainty quantification and risk analysis.

## Features

### Core Capabilities
- **Multiple ML Models**: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVM, and LSTM neural networks
- **Hyperparameter Tuning**: Automated optimization using Grid Search and Bayesian optimization
- **Confidence Intervals**: Uncertainty quantification for all predictions using Monte Carlo methods and residual analysis
- **Technical Indicators**: 25+ features including MACD, RSI, Bollinger Bands, moving averages, and volatility measures
- **Risk Analysis**: Comprehensive risk metrics and trading recommendations
- **Interactive Visualizations**: Historical and future predictions with confidence bands

### Advanced Features
- **Ensemble Predictions**: Weighted combination of multiple models for improved accuracy
- **Walk-Forward Validation**: Time-series aware model evaluation
- **Trading Strategy Simulation**: Backtesting with performance metrics
- **Future Price Forecasting**: Multi-day predictions with uncertainty bounds

## Installation

### Requirements
```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
```

### Optional Dependencies (for enhanced features)
```bash
pip install scikit-optimize scikeras  # For Bayesian optimization
```

### Quick Install
```bash
git clone https://github.com/TeyvenN/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from corrected_stock_predictor import CorrectedStockPredictor

# Initialize predictor
predictor = CorrectedStockPredictor(
    symbol="AAPL", 
    period="2y", 
    tune_hyperparameters=True
)

# Fetch and train
predictor.fetch_data()
predictor.train_models()

# Evaluate models
evaluation = predictor.evaluate_models()
print(evaluation)

# Plot predictions with confidence intervals
predictor.plot_future_predictions_with_confidence(days=14)
```

### Advanced Usage

```python
# Risk analysis
risk_analysis = predictor.analyze_prediction_risk(days=7)

# Trading recommendations
predictor.generate_trading_recommendations()

# Custom confidence levels
future_pred = predictor.predict_future_with_confidence(
    days=30, 
    confidence_level=0.99
)
```

## Model Performance

The system typically achieves:
- **R² Score**: 0.6-0.8 for best models
- **Directional Accuracy**: 52-60% (above random chance)
- **RMSE**: 3-9% of stock price, depending on volatility
- **Risk-Adjusted Returns**: Positive Sharpe ratios in backtesting

### Model Comparison Example
```
Model                 R²      RMSE    Directional Accuracy
Ridge                0.86     3.71    54.76
Linear               0.85     3.73    52.38
Random Forest        0.23     8.55    53.57
LSTM                 0.19     8.80    60.71%
Ensemble             0.82     4.18    52.38%
```

## Technical Indicators Used

### Price-Based Features
- **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)
- **Price Ratios**: Close/SMA ratios, momentum indicators
- **Returns**: 1-day, 3-day, 5-day percentage changes

### Technical Analysis
- **MACD**: Signal line, histogram
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: Position, width, squeeze indicators
- **ATR**: Average True Range for volatility

### Volume Analysis
- **Volume Ratios**: Current vs. historical averages
- **Volume Momentum**: Multi-period volume changes

## Confidence Intervals

### Uncertainty Estimation Methods

#### LSTM Models
- **Monte Carlo Dropout**: Samples from model uncertainty during inference
- **Temporal Uncertainty**: Accounts for increasing uncertainty over time

#### Traditional ML Models
- **Residual-Based**: Uses historical prediction errors
- **Time-Decay**: Uncertainty compounds over prediction horizon

#### Ensemble Models
- **Variance Weighting**: Combines individual model uncertainties
- **Model Disagreement**: Accounts for prediction divergence

### Risk Metrics
- **Value at Risk (VaR)**: Potential losses at confidence levels
- **Maximum Drawdown**: Worst-case scenario analysis
- **Probability of Loss**: Likelihood of negative returns
- **Risk/Reward Ratios**: Expected gain vs. potential loss

## Configuration Options

### Model Parameters
```python
predictor = CorrectedStockPredictor(
    symbol="AAPL",                    # Stock ticker
    period="2y",                      # Data period
    tune_hyperparameters=True         # Enable optimization
)
```

### Prediction Parameters
```python
# Future predictions
predictor.predict_future_with_confidence(
    days=14,                          # Forecast horizon
    confidence_level=0.95             # Confidence interval
)
```

### Visualization Parameters
```python
# Plot settings
predictor.plot_future_predictions_with_confidence(
    days=14,                          # Days to forecast
    confidence_level=0.95             # Confidence band
)
```


## API Reference

### Core Methods

#### `CorrectedStockPredictor`
Main prediction class with hyperparameter tuning and confidence intervals.

**Parameters:**
- `symbol` (str): Stock ticker symbol
- `period` (str): Data period ('1y', '2y', '5y', etc.)
- `tune_hyperparameters` (bool): Enable automated optimization

#### `fetch_data()`
Download stock data from Yahoo Finance.

**Returns:** pandas.DataFrame with OHLCV data

#### `train_models()`
Train all models with hyperparameter optimization.

#### `predict_future_with_confidence(days, confidence_level)`
Generate future predictions with uncertainty bounds.

**Parameters:**
- `days` (int): Number of days to predict
- `confidence_level` (float): Confidence interval (0.90, 0.95, 0.99)

**Returns:** Dictionary with predictions, lower_bound, upper_bound for each model

#### `analyze_prediction_risk(days)`
Comprehensive risk analysis of predictions.

**Returns:** Dictionary with risk metrics for each model

### Visualization Methods

#### `plot_future_predictions_with_confidence(days, confidence_level)`
Interactive plots with confidence bands.

#### `plot_predictions(days_to_show)`
Historical prediction accuracy visualization.

## Performance Optimization

### Speed vs. Accuracy Trade-offs
```python
# Fast execution (reduced tuning)
predictor = CorrectedStockPredictor(
    symbol="AAPL",
    period="1y",
    tune_hyperparameters=False
)

# High accuracy (full optimization)
predictor = CorrectedStockPredictor(
    symbol="AAPL",
    period="2y", 
    tune_hyperparameters=True
)
```

### Memory Usage
- **Dataset Size**: ~500-1000 rows typical for 2-year period
- **Feature Count**: 28 engineered features per model
- **Memory Requirements**: ~100MB for full analysis

## Limitations and Disclaimers

### Model Limitations
- **Market Regime Changes**: Models may underperform during unusual market conditions
- **Black Swan Events**: Confidence intervals don't account for extreme rare events
- **Data Dependency**: Requires sufficient historical data (minimum 1 year recommended)

### Financial Disclaimers
- **Not Financial Advice**: This tool is for educational and research purposes only
- **Risk Warning**: Stock trading involves substantial risk of loss
- **Past Performance**: Historical results don't guarantee future performance
- **Professional Consultation**: Always consult with financial professionals before making investment decisions

### Technical Limitations
- **Prediction Horizon**: Accuracy decreases significantly beyond 2-3 weeks
- **Feature Engineering**: Limited to technical indicators (no fundamental analysis)
- **Market Hours**: Only accounts for regular trading session data

## Contributing

### Development Setup
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for all public methods
- Include type hints where appropriate

### Contribution Areas
- Additional technical indicators
- Alternative uncertainty quantification methods
- Performance optimization
- Extended validation frameworks
- New visualization options

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stock_price_prediction,
  title={Stock Price Prediction with Confidence Intervals},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/stock-price-prediction}
}
```

## Support

### Documentation
- [API Reference](docs/api.md)
- [Tutorial Notebooks](examples/)
- [FAQ](docs/faq.md)

### Issues and Questions
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Use GitHub Discussions
- **Questions**: Stack Overflow with `stock-prediction` tag

### Contact
- **Email**: ndlovuteyven@gmail.com
- **GitHub**: [@TeyveN](https://github.com/yourusername)

## Changelog

### v1.0.0 (2025-01-01)
- Initial release with confidence intervals
- Hyperparameter tuning implementation
- Risk analysis framework
- Interactive visualizations

### v0.9.0 (2024-12-15)
- Beta release with core prediction models
- Basic uncertainty quantification
- Historical backtesting capabilities

---

**Disclaimer**: This software is provided "as is" without warranty. Users assume full responsibility for any financial decisions made using this tool.
# Stock Price Prediction with Confidence Intervals

A comprehensive Python system for predicting stock prices using multiple machine learning models with uncertainty quantification and risk analysis.

## Features

### Core Capabilities
- **Multiple ML Models**: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVM, and LSTM neural networks
- **Hyperparameter Tuning**: Automated optimization using Grid Search and Bayesian optimization
- **Confidence Intervals**: Uncertainty quantification for all predictions using Monte Carlo methods and residual analysis
- **Technical Indicators**: 25+ features including MACD, RSI, Bollinger Bands, moving averages, and volatility measures
- **Risk Analysis**: Comprehensive risk metrics and trading recommendations
- **Interactive Visualizations**: Historical and future predictions with confidence bands

### Advanced Features
- **Ensemble Predictions**: Weighted combination of multiple models for improved accuracy
- **Walk-Forward Validation**: Time-series aware model evaluation
- **Trading Strategy Simulation**: Backtesting with performance metrics
- **Future Price Forecasting**: Multi-day predictions with uncertainty bounds

## Installation

### Requirements
```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
```

### Optional Dependencies (for enhanced features)
```bash
pip install scikit-optimize scikeras  # For Bayesian optimization
```

### Quick Install
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from corrected_stock_predictor import CorrectedStockPredictor

# Initialize predictor
predictor = CorrectedStockPredictor(
    symbol="AAPL", 
    period="2y", 
    tune_hyperparameters=True
)

# Fetch and train
predictor.fetch_data()
predictor.train_models()

# Evaluate models
evaluation = predictor.evaluate_models()
print(evaluation)

# Plot predictions with confidence intervals
predictor.plot_future_predictions_with_confidence(days=14)
```

### Advanced Usage

```python
# Risk analysis
risk_analysis = predictor.analyze_prediction_risk(days=7)

# Trading recommendations
predictor.generate_trading_recommendations()

# Custom confidence levels
future_pred = predictor.predict_future_with_confidence(
    days=30, 
    confidence_level=0.99
)
```

## Model Performance

The system typically achieves:
- **R² Score**: 0.6-0.8 for best models
- **Directional Accuracy**: 52-60% (above random chance)
- **RMSE**: 3-9% of stock price, depending on volatility
- **Risk-Adjusted Returns**: Positive Sharpe ratios in backtesting

### Model Comparison Example
```
Model                 R²      RMSE    Directional Accuracy
Ridge                0.86     3.71    54.76
Linear               0.85     3.73    52.38
Random Forest        0.23     8.55    53.57
LSTM                 0.19     8.80    60.71%
Ensemble             0.82     4.18    52.38%
```

## Technical Indicators Used

### Price-Based Features
- **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)
- **Price Ratios**: Close/SMA ratios, momentum indicators
- **Returns**: 1-day, 3-day, 5-day percentage changes

### Technical Analysis
- **MACD**: Signal line, histogram
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: Position, width, squeeze indicators
- **ATR**: Average True Range for volatility

### Volume Analysis
- **Volume Ratios**: Current vs. historical averages
- **Volume Momentum**: Multi-period volume changes

## Confidence Intervals

### Uncertainty Estimation Methods

#### LSTM Models
- **Monte Carlo Dropout**: Samples from model uncertainty during inference
- **Temporal Uncertainty**: Accounts for increasing uncertainty over time

#### Traditional ML Models
- **Residual-Based**: Uses historical prediction errors
- **Time-Decay**: Uncertainty compounds over prediction horizon

#### Ensemble Models
- **Variance Weighting**: Combines individual model uncertainties
- **Model Disagreement**: Accounts for prediction divergence

### Risk Metrics
- **Value at Risk (VaR)**: Potential losses at confidence levels
- **Maximum Drawdown**: Worst-case scenario analysis
- **Probability of Loss**: Likelihood of negative returns
- **Risk/Reward Ratios**: Expected gain vs. potential loss

## Configuration Options

### Model Parameters
```python
predictor = CorrectedStockPredictor(
    symbol="AAPL",                    # Stock ticker
    period="2y",                      # Data period
    tune_hyperparameters=True         # Enable optimization
)
```

### Prediction Parameters
```python
# Future predictions
predictor.predict_future_with_confidence(
    days=14,                          # Forecast horizon
    confidence_level=0.95             # Confidence interval
)
```

### Visualization Parameters
```python
# Plot settings
predictor.plot_future_predictions_with_confidence(
    days=14,                          # Days to forecast
    confidence_level=0.95             # Confidence band
)
```


## API Reference

### Core Methods

#### `CorrectedStockPredictor`
Main prediction class with hyperparameter tuning and confidence intervals.

**Parameters:**
- `symbol` (str): Stock ticker symbol
- `period` (str): Data period ('1y', '2y', '5y', etc.)
- `tune_hyperparameters` (bool): Enable automated optimization

#### `fetch_data()`
Download stock data from Yahoo Finance.

**Returns:** pandas.DataFrame with OHLCV data

#### `train_models()`
Train all models with hyperparameter optimization.

#### `predict_future_with_confidence(days, confidence_level)`
Generate future predictions with uncertainty bounds.

**Parameters:**
- `days` (int): Number of days to predict
- `confidence_level` (float): Confidence interval (0.90, 0.95, 0.99)

**Returns:** Dictionary with predictions, lower_bound, upper_bound for each model

#### `analyze_prediction_risk(days)`
Comprehensive risk analysis of predictions.

**Returns:** Dictionary with risk metrics for each model

### Visualization Methods

#### `plot_future_predictions_with_confidence(days, confidence_level)`
Interactive plots with confidence bands.

#### `plot_predictions(days_to_show)`
Historical prediction accuracy visualization.

## Performance Optimization

### Speed vs. Accuracy Trade-offs
```python
# Fast execution (reduced tuning)
predictor = CorrectedStockPredictor(
    symbol="AAPL",
    period="1y",
    tune_hyperparameters=False
)

# High accuracy (full optimization)
predictor = CorrectedStockPredictor(
    symbol="AAPL",
    period="2y", 
    tune_hyperparameters=True
)
```

### Memory Usage
- **Dataset Size**: ~500-1000 rows typical for 2-year period
- **Feature Count**: 28 engineered features per model
- **Memory Requirements**: ~100MB for full analysis

## Limitations and Disclaimers

### Model Limitations
- **Market Regime Changes**: Models may underperform during unusual market conditions
- **Black Swan Events**: Confidence intervals don't account for extreme rare events
- **Data Dependency**: Requires sufficient historical data (minimum 1 year recommended)

### Financial Disclaimers
- **Not Financial Advice**: This tool is for educational and research purposes only
- **Risk Warning**: Stock trading involves substantial risk of loss
- **Past Performance**: Historical results don't guarantee future performance
- **Professional Consultation**: Always consult with financial professionals before making investment decisions

### Technical Limitations
- **Prediction Horizon**: Accuracy decreases significantly beyond 2-3 weeks
- **Feature Engineering**: Limited to technical indicators (no fundamental analysis)
- **Market Hours**: Only accounts for regular trading session data

## Contributing

### Development Setup
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for all public methods
- Include type hints where appropriate

### Contribution Areas
- Additional technical indicators
- Alternative uncertainty quantification methods
- Performance optimization
- Extended validation frameworks
- New visualization options



## Citation

If you use this code in your research, please cite:

```bibtex
@software{stock_price_prediction,
  title={Stock Price Prediction with Confidence Intervals},
  author={Teyven Ndlovu},
  year={2025},
  url={https://github.com/yourusername/stock-price-prediction}
}
```

## Support

### Documentation
- [API Reference](docs/api.md)
- [Tutorial Notebooks](examples/)
- [FAQ](docs/faq.md)

### Issues and Questions
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Use GitHub Discussions
- **Questions**: Stack Overflow with `stock-prediction` tag

### Contact
- **Email**: ndlovuteyven@gmail.com
- **GitHub**: [@TeyveN](https://github.com/TeyvenN)

## Changelog

### v1.0.0 (2025-01-01)
- Initial release with confidence intervals
- Hyperparameter tuning implementation
- Risk analysis framework
- Interactive visualizations

### v0.9.0 (2024-12-15)
- Beta release with core prediction models
- Basic uncertainty quantification
- Historical backtesting capabilities

---

**Disclaimer**: This software is provided "as is" without warranty. Users assume full responsibility for any financial decisions made using this tool.
