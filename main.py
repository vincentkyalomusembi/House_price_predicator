import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def generate_house_data(n_samples=100):
    """Generate synthetic house data for training"""
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    """Train a linear regression model on house data"""
    df = generate_house_data(n_samples=100)
    X = df[['size']]  # Feature needs to be 2D
    y = df['price']   # Target variable
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Train the model once when the app starts
@st.cache_resource
def get_trained_model():
    return train_model()

def create_visualization(model, user_size, user_prediction):
    """Create a visualization showing model performance and user's prediction"""
    # Generate sample data for visualization
    df = generate_house_data(n_samples=100)
    X = df[['size']]
    y = df['price']
    
    # Split data to show training vs test performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data
    ax.scatter(X_train['size'], y_train, color='lightblue', alpha=0.6, label='Training data', s=30)
    
    # Plot test data (actual vs predicted)
    ax.scatter(X_test['size'], y_test, color='blue', alpha=0.7, label='Actual test prices', s=40)
    ax.scatter(X_test['size'], y_pred, color='red', alpha=0.7, label='Predicted test prices', s=40)
    
    # Highlight user's prediction
    ax.scatter(user_size, user_prediction, color='green', s=200, marker='*', 
               label=f'Your prediction ({user_size} sq ft)', edgecolor='black', linewidth=2)
    
    # Add trend line
    size_range = np.linspace(X['size'].min(), X['size'].max(), 100)
    trend_predictions = model.predict(size_range.reshape(-1, 1))
    ax.plot(size_range, trend_predictions, color='red', linestyle='--', alpha=0.8, label='Model trend line')
    
    # Customize the plot
    ax.set_xlabel('House Size (sq ft)')
    ax.set_ylabel('Price ($)')
    ax.set_title('House Price Prediction Model Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show prices nicely
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    return fig

st.header("Simple Linear Regression House Price Prediction App")
st.write("Enter your house size to predict its price")

# Get the trained model
model = get_trained_model()

# User input
size = st.number_input('House size (sq ft)', min_value=500, max_value=2000, value=1500)

if st.button('Predict Price'):
    # Make prediction
    predicted_price = model.predict([[size]])
    st.success(f"Estimated price: ${predicted_price[0]:,.2f}")
    
    # Show some additional info
    st.info(f"For a house of {size} sq ft, the predicted price is ${predicted_price[0]:,.2f}")
    
    # Create and display visualization
    st.subheader("Model Visualization")
    fig = create_visualization(model, size, predicted_price[0])
    st.pyplot(fig)
    
    # Optional: Show model details
    with st.expander("Model Details"):
        st.write(f"Model coefficient (price per sq ft): ${model.coef_[0]:.2f}")
        st.write(f"Model intercept: ${model.intercept_:.2f}")
        st.write("The green star shows your prediction on the model!")
        st.write("Blue dots are actual prices, red dots are model predictions on test data.")
    
    # Show visualization
    fig = create_visualization(model, size, predicted_price[0])
    st.pyplot(fig)