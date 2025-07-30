# 🏠 House Price Predictor

A machine learning project for predicting house prices using various algorithms and features. This repository contains multiple implementations of house price prediction models with different approaches, from simple linear regression to more complex multi-feature models.

## ✨ Features

- **Multiple Prediction Models**: Various implementations using different approaches
- **Interactive Web Apps**: Streamlit-based user interfaces for easy interaction
- **Real Dataset Support**: Works with real house price datasets from India
- **Synthetic Data Generation**: Ability to generate synthetic house data for testing
- **Visualization**: Interactive charts and plots to visualize predictions and model performance
- **Pre-trained Models**: Saved models ready for immediate use
- **Multiple Input Features**: Support for various house characteristics (size, bedrooms, bathrooms, etc.)

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **XGBoost & LightGBM** - Advanced ML algorithms
- **Joblib** - Model persistence

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vincentkyalomusembi/House_price_predicator.git
   cd House_price_predicator
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

This project includes multiple Streamlit applications with different approaches:

### 1. Main App - Simple Linear Regression (`main.py`)
A basic house price predictor using synthetic data and linear regression.

```bash
streamlit run main.py
```

**Features:**
- Simple size-to-price prediction
- Synthetic data generation
- Interactive visualization with matplotlib
- Model performance metrics

### 2. Advanced App - Multi-feature Model (`front.py`)
A more sophisticated model using multiple house features.

```bash
streamlit run front.py
```

**Features:**
- Multiple input features (bedrooms, bathrooms, living area, condition, schools nearby)
- Uses pre-trained model from `notebook/model.pkl`
- Balloon animation on prediction

### 3. Saved Model App (`app_saved_model.py`)
Simple app using a pre-trained model with area and bedroom inputs.

```bash
streamlit run app_saved_model.py
```

**Features:**
- Area and bedroom-based prediction
- Uses `house_price_model.pkl`
- Clean, minimalist interface

### 4. Retrain App (`app_retrain.py`)
Trains model on-the-fly with interactive Plotly visualizations.

```bash
streamlit run app_retrain.py
```

**Features:**
- Real-time model training
- Interactive Plotly charts
- Dynamic prediction highlighting

## 📁 Project Structure

```
House_price_predicator/
├── Dataset/                    # House price datasets
│   ├── House Price India.csv   # Indian house price data
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   └── sample_submission.csv   # Sample submission format
├── notebook/                   # Jupyter notebooks and models
│   ├── house.ipynb            # House price analysis notebook
│   ├── house_price.ipynb      # Price prediction notebook
│   ├── india.ipynb            # India dataset analysis
│   ├── model.pkl              # Trained model for front.py
│   └── house_price_model.pkl   # Model for saved model app
├── main.py                    # Main Streamlit app (simple regression)
├── front.py                   # Advanced multi-feature app
├── app_saved_model.py         # Simple saved model app
├── app_retrain.py             # Retrain model app
├── house_price_model.pkl      # Pre-trained model
├── learn.txt                  # Learning/example code
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📊 Dataset Information

The project includes several datasets:

- **House Price India.csv**: Real house price data from India with various features
- **train.csv / test.csv**: Training and testing datasets for model development
- **Synthetic Data**: Generated programmatically for demonstration purposes

### Dataset Features (varies by dataset):
- House size/area (square feet)
- Number of bedrooms
- Number of bathrooms
- Living area
- House condition
- Number of nearby schools
- Location-based features

## 🔧 Model Information

The project includes several pre-trained models:

1. **Linear Regression Models**: Simple models for basic predictions
2. **Multi-feature Models**: Complex models using multiple house characteristics
3. **Ensemble Models**: Advanced algorithms like XGBoost and LightGBM

Models are saved using:
- **Pickle**: For scikit-learn models
- **Joblib**: For efficient model persistence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 Usage Examples

### Basic Prediction
```python
import pickle
import numpy as np

# Load model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make prediction
features = np.array([[2000, 3]])  # [area, bedrooms]
price = model.predict(features)
print(f"Predicted price: ${price[0]:,.2f}")
```

### Running Jupyter Notebooks
```bash
jupyter notebook
# Navigate to notebook/ directory and open desired .ipynb file
```

## 🔍 Model Performance

The models provide price predictions based on various house features. Performance varies by model complexity and dataset used. Check individual notebook files for detailed performance metrics and analysis.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
