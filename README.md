# Stock-Direction-Prediction
Stock Prediction App Usage and Installation Guide
This guide provides step-by-step instructions for installing and using the Stock Prediction Web Application, which predicts the next day's stock price direction (Up or Down) for a given ticker symbol using historical data and machine learning.

______________________________________________________________________________________________________________________________________________________________

## Installation Guide
### Prerequisites
Before installing the application, ensure you have the following installed on your system:
- Python 3.8 or higher: Download and install from python.org.
- pip: Python's package manager, typically included with Python.
- A modern web browser (e.g., Chrome, Firefox, Edge).
- An active internet connection to download dependencies and fetch stock data.

### Step-by-Step Installation

#### 1. Clone or Download the Application Code

- Save the provided Python script as server.py in a dedicated project folder (e.g., stock_prediction_app).

- Alternatively, if the code is hosted in a repository, clone it using:
  - git clone <repository-url>
  - cd stock_prediction_app

#### 2. Create a Virtual Environment (Optional but Recommended)

- Create a virtual environment to isolate dependencies:
  - python -m venv venv
- Activate the virtual environment:

- On Windows:
  - venv\Scripts\activate
- On macOS/Linux:
  - source venv/bin/activate

#### 3. Install Required Python Packages
- Install the necessary libraries using pip:
  - pip install pandas numpy matplotlib yfinance ta seaborn scikit-learn xgboost joblib flask waitress
  - or pip install -r requirements.txt 
- This command installs all dependencies required for data processing, machine learning, and running the web server.

#### 4. Set Up HTML Templates
- Create a folder named templates in the same directory as server.py.
- Add the following three HTML files inside the templates folder:
  - index.html
  - result.html
  - error.html
- Verify Directory Structure
- Your project folder should look like this:
<pre>
  stock_prediction_app/
  ├── static
  |      ├── index.css
  |      ├── index.css
  |      ├── index.css
  ├── templates/
  │   ├── index.html
  │   ├── result.html
  │   ├── error.html
  ├── server.py
  ├── training.py
  └── venv/ (if using virtual environment)
</pre>
  
#### 5. Run the Application
- From the project directory, run the application:
- python server.py
- The application uses waitress to serve the Flask app on http://localhost:8000 by default.
- Access the Application
- Open a web browser and navigate to http://localhost:8000.
- You should see the stock prediction home page.
__________________________________________________________________________________________________________________________________________________________________

## Usage Guide
### Overview
The Stock Prediction Web Application allows users to input a stock ticker symbol (e.g., AAPL for Apple Inc.) and receive a prediction of whether the stock's price is likely to go Up or Down the next trading day, along with the probability of the prediction, the latest closing price, and the percentage change.

### Steps to Use the Application

#### 1. Access the Home Page
- Navigate to http://localhost:8000 in your web browser.
- You will see a simple form prompting you to enter a stock ticker symbol.

#### 2. Enter a Stock Ticker
- Input a valid stock ticker symbol (e.g., AAPL for Apple, TSLA for Tesla, MSFT for Microsoft).
- Ticker symbols are typically uppercase and correspond to stocks listed on major exchanges (e.g., NYSE, NASDAQ).
- Click the Predict button to submit the ticker.

#### 3. View the Prediction
- If the ticker is valid, the application will:
- Fetch 5 years of historical stock data for the ticker, along with VIX and S&P 500 data.
- Calculate technical indicators (e.g., EMA, RSI, MACD, Bollinger Bands, VWAP, ATR, OBV).
- Train an XGBoost model to predict the next day's price direction.
- Display the prediction results, including:
  - Company Name: The full name of the company (e.g., Apple Inc.).
  - Today's Date: The current date.
  - Latest Close Price: The most recent closing price of the stock.
  - Latest Change: The percentage change in the stock price from the previous day.
  - Predicted Direction: Whether the stock is predicted to go Up or Down tomorrow.
  - Probability: The confidence level of the prediction (as a percentage).
  - Prediction Date: The date for which the prediction applies (typically the next trading day).

#### 4. Handle Errors
- If you enter an invalid ticker (e.g., a non-existent symbol like ZZZZ), the application will display an error message: "Invalid ticker symbol. Please try again."
- Click the Back to Home link to try again.

#### 5. Repeat for Other Stocks
- Return to the home page by clicking Back to Home and enter another ticker to get a new prediction.

#### Example Usage
- Input: Enter AAPL and click Predict.
- Output: The result page might show:
  - Company: Apple Inc.
  - Today's Date: 2025-04-14
  - Latest Close Price: $150.25
  - Latest Change: +1.20%
  - Predicted Direction for 2025-04-15: Up
  - Probability: 75.50%
  
#### Notes
- Data Source: The application uses the yfinance library to fetch historical stock data, VIX (^VIX), and S&P 500 (^GSPC) data from Yahoo Finance.
- Prediction Model: The application trains an XGBoost classifier on technical indicators and historical data to predict price direction.
- Limitations:
  - Predictions are based on historical data and technical indicators, not guaranteed outcomes.
  - The model assumes market conditions remain consistent with historical patterns.
  - Only one ticker can be predicted at a time.
- Performance: The application prints the XGBoost model's accuracy on test data to the console (e.g., "XGBoost Directional Accuracy: 62.50%").
__________________________________________________________________________________________________________________________________________________________________

## Troubleshooting
- Error: "Invalid ticker symbol"
  - Ensure the ticker is correct and corresponds to a publicly traded company (e.g., AAPL, not APPLE).
  - Check your internet connection, as data is fetched from Yahoo Finance.
- Error: "ModuleNotFoundError"
  - Verify that all required packages are installed (run pip install -r requirements.txt if you have a requirements.txt file).
  - Ensure you are in the correct virtual environment if using one.
- Application Not Loading
  - Confirm that the Flask server is running (python app.py).
  - Check that port 8000 is not blocked by another application.
  - Try accessing http://127.0.0.1:8000 instead of localhost.
- Missing Templates
  - Ensure the templates folder contains index.html, result.html, and error.html with the exact content provided.
__________________________________________________________________________________________________________________________________________________________________

## Additional Information
- Dependencies:
  - pandas, numpy: Data manipulation.
  - matplotlib, seaborn: Data visualization (used internally).
  - yfinance: Fetching stock data.
  - ta: Technical analysis indicators.
  - scikit-learn: Machine learning utilities.
  - xgboost: Gradient boosting model.
  - joblib: Model persistence.
  - flask, waitress: Web server framework and production server.
- Port Configuration:
  - The application runs on port 8000 by default. To change the port, modify the port parameter in the serve(app, host='0.0.0.0', port=8000) line in app.py.
- Extending the Application:
  - To add more features (e.g., additional indicators, different models), modify the fetch_and_preprocess_data function in app.py.
  - To improve the UI, update the HTML templates in the templates folder with custom CSS or JavaScript.
- Shutting Down:
  - Stop the server by pressing Ctrl+C in the terminal where app.py is running.
 __________________________________________________________________________________________________________________________________________________________________

This guide should help you successfully install and use the Stock Prediction Web Application. For further assistance, consult the documentation of the libraries used or reach out to the developer.

