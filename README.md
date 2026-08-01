# Time Series Forecasting Toolkit

An interactive **Streamlit application** for exploring, forecasting and backtesting time-series data.

The toolkit helps analysts and data scientists compare forecasting approaches, test model performance on historical data and interpret forecast accuracy without rebuilding analysis pipelines.

> The focus is not only on generating forecasts, but on understanding when they can be trusted.

## Live app

Add the deployed Streamlit URL here:

```text
https://YOUR-APP-NAME.streamlit.app
```

## Features

- Load repository data, local files, URLs or live currency rates
- Select date and value columns interactively
- Filter datasets containing multiple time series
- Replace zeroes with missing values when appropriate
- Resample to days, business days, weeks, months or years
- Compare simple, double and triple exponential smoothing, auto-ARIMA and Prophet
- Forecast future values or backtest against held-out historical observations
- Evaluate backtests using RMSE and normalised percentage difference (NPD)
- Zoom the final plot by choosing how many historical observations to display
- Adjust the y-axis automatically or manually
- Download forecast results as CSV
- View plain-English interpretation of methods and performance

## Why this tool?

Forecasting models can appear accurate in-sample but perform poorly when conditions change. This application makes that gap visible through built-in backtesting and error analysis.

Backtesting reserves the most recent observations, fits the model only on earlier data, and compares the predictions with values the model did not see during training.

## Repository structure

```text
.
├── app
│   └── time_series_app.py
├── data
├── notebooks
│   └── TS.ipynb
├── src
│   ├── __init__.py
│   └── forecasting.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The original notebook is retained in `notebooks/TS.ipynb` to document the development process and underlying methodology. The Streamlit application is the main interface.

## Running locally

```bash
git clone https://github.com/steviecurran/time-series-toolkit.git
cd time-series-toolkit

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app/time_series_app.py
```

## Using the application

1. Choose a data source.
2. Select the date and value columns.
3. Optionally filter the dataset to one category or series.
4. Choose whether to resample the dates.
5. Select a forecasting method and forecast horizon.
6. Choose **Forecast** or **Test**.
7. Run the model.
8. Adjust the final-plot zoom without refitting the model.
9. Review the accuracy measures and download the results.

## Optional filtering

Filtering is useful when one file contains several distinct time series.

For example, `all_stocks_5yr+names.csv` contains historical prices for many companies. Selecting a company isolates that company's observations before the series is prepared and modelled.

## Forecasting methods

### Simple exponential smoothing

Best suited to relatively stable series without a clear trend or seasonal structure.

### Double exponential smoothing

Adds a trend component and is useful when values systematically increase or decrease over time.

### Triple exponential smoothing

Also known as Holt-Winters. Adds both trend and seasonality.

### Auto-ARIMA

Automatically searches for an ARIMA specification using the observed autocorrelation structure.

### Prophet

Designed for changing trends, missing observations and optional national-holiday effects.

## Forecast evaluation

### RMSE

Root mean squared error measures prediction error in the same units as the original series:

$ \mathrm{RMSE} =\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}$

Lower values indicate smaller errors, but the value must be interpreted relative to the scale of the data.

### NPD

The normalised percentage difference expresses RMSE relative to the mean absolute observed value during the test period:

$\mathrm{NPD}=100\times\frac{\mathrm{RMSE}}{\mathrm{mean}(|y_{\mathrm{actual}}|)}$

The app uses this broad interpretation:

| NPD | Interpretation |
|---:|---|
| Below 5% | Excellent |
| 5% to below 10% | Good |
| 10% to below 20% | Reasonable |
| 20% or more | Poor |

These are heuristic labels rather than universal performance thresholds.

## Example 1: Air passengers

`AirPassengers.csv` provides a monthly series with trend and seasonality. A typical workflow is to compare Holt-Winters, ARIMA and Prophet, backtest the latest 12 months, and inspect RMSE and NPD.

## Example 2: Stock prices

`all_stocks_5yr+names.csv` contains historical prices for many companies. Use the optional filter to select one company and resample to business days if required.

This example illustrates an important limitation: a model can fit historical patterns well but still fail during sudden market changes.

## Example 3: Currency rates

The currency option downloads recent exchange-rate observations. Plot labels identify the rate explicitly, for example:

```text
NZD per GBP
```

The requested calendar period may contain fewer observations because weekends and market holidays are excluded.

## Dependencies

- Streamlit
- NumPy
- pandas
- Matplotlib
- statsmodels
- yfinance
- pmdarima
- Prophet

`pmdarima` contains compiled extensions. Use the Python and NumPy versions specified in `requirements.txt`.

## Limitations

- Forecasts are not guarantees and may fail during structural breaks.
- Backtest results depend on the selected horizon and historical period.
- Interpolation during resampling can introduce synthetic observations.
- The heuristic NPD labels should not replace domain-specific acceptance criteria.
- No rolling-origin cross-validation is currently implemented.

## Future improvements

- Rolling-origin cross-validation
- Forecast intervals
- Additional baseline models
- Model-comparison table
- Residual diagnostics
- Automated tests
- Continuous integration

## Licence

This project is released under the MIT Licence.
