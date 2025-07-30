# 🏠 House Price Predictor

A machine learning project that predicts house prices using multiple algorithms and provides an interactive Streamlit web application for real-time predictions.

## 📋 Project Overview

This project implements and compares different machine learning models to predict house prices based on various features like number of bedrooms, bathrooms, living area, house condition, and nearby amenities. The project includes both synthetic data generation for learning purposes and real-world Indian housing market data analysis.

## 🚀 Features

- **Multiple ML Models**: Decision Tree, Linear Regression, and Random Forest
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Data Visualization**: Comprehensive plots showing model performance
- **Model Comparison**: Evaluation metrics to compare different algorithms
- **Real Dataset**: Analysis of Indian housing market data
- **Synthetic Data**: Generated data for learning and experimentation

## 📁 Project Structure

```
House-price-predicator/
├── Dataset/
│   ├── House Price India.csv    # Real Indian housing data
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── sample_submission.csv   # Sample submission format
├── notebook/
│   ├── house_price.ipynb       # Synthetic data ML tutorial
│   ├── india.ipynb             # Real data analysis
│   └── model.pkl              # Saved trained model
├── main.py                     # Streamlit web application
├── app_retrain.py             # Model retraining utilities
├── app_saved_model.py         # Saved model utilities
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vincentkyalomusembi/House_price_predicator.git
   cd House_price_predicator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Web Application
Run the interactive Streamlit app:
```bash
streamlit run main.py
```

The app will open in your browser where you can:
- Input house size (square feet)
- Get instant price predictions
- View model performance visualization
- Explore model details and coefficients

### Jupyter Notebooks
Explore the analysis notebooks:

1. **Synthetic Data Tutorial** (`notebook/house_price.ipynb`):
   - Learn linear regression basics
   - Generate synthetic house data
   - Train and visualize simple models

2. **Real Data Analysis** (`notebook/india.ipynb`):
   - Comprehensive data analysis
   - Multiple feature engineering
   - Advanced model comparison

## 📊 Models & Performance

### Models Implemented:
1. **Linear Regression**: Simple baseline model
2. **Decision Tree Regressor**: Non-linear relationships
3. **Random Forest Regressor**: Ensemble method (best performer)

### Features Used:
- Number of bedrooms
- Number of bathrooms  
- Living area (sq ft)
- House condition
- Number of schools nearby

### Model Evaluation:
- **Random Forest**: Lowest Mean Absolute Error
- **Decision Tree**: Good performance with hyperparameter tuning
- **Linear Regression**: Simple but effective baseline

## 📈 Key Insights

- **House condition** significantly impacts price
- **Living area** shows strong positive correlation with price
- **Number of schools nearby** affects property value
- **Random Forest** performs best due to feature interactions

## 🔧 Technical Details

### Dependencies:
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Joblib**: Model persistence

### Machine Learning Pipeline:
1. Data loading and preprocessing
2. Feature selection and engineering
3. Train-test split (80-20)
4. Hyperparameter tuning with GridSearchCV
5. Model training and evaluation
6. Performance comparison

## 📚 Learning Objectives

This project demonstrates:
- **Data Science Workflow**: From data exploration to deployment
- **ML Model Comparison**: Understanding different algorithms
- **Web Development**: Creating interactive ML applications
- **Model Evaluation**: Using proper metrics and validation
- **Best Practices**: Code organization and documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Vincent Kyalo Musembi**
- GitHub: [@vincentkyalomusembi](https://github.com/vincentkyalomusembi)

## 🔮 Future Enhancements

- [ ] Add more sophisticated models (XGBoost, Neural Networks)
- [ ] Implement feature importance analysis
- [ ] Add model explainability (SHAP values)
- [ ] Create API endpoints for external integration
- [ ] Add more interactive visualizations
- [ ] Implement real-time data updates

## 📞 Support

If you have any questions or need help with the project, please:
1. Check the [Issues](https://github.com/vincentkyalomusembi/House_price_predicator/issues) section
2. Create a new issue if your question isn't already addressed
3. Provide detailed information about your problem

---

⭐ **If this project helped you learn something new, please give it a star!** ⭐