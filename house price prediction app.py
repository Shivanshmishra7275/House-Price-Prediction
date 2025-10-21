import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .prediction-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏠 House Price Prediction App</h1>', unsafe_allow_html=True)
st.markdown("### Predict house prices using machine learning based on property features")

# Load or train model
@st.cache_data
def load_data():
    """Load the house price dataset"""
    try:
        # Try to load from the path in your notebook
        dataset = pd.read_csv("house_prices_8000.csv")
        return dataset
    except FileNotFoundError:
        st.error("Dataset file 'house_prices_8000.csv' not found. Please ensure the file is in the same directory as this app.")
        return None

@st.cache_resource
def train_model():
    """Train the machine learning model"""
    dataset = load_data()
    if dataset is None:
        return None, None
    
    # Prepare features and target
    X = dataset.drop('price', axis=1)
    X = pd.get_dummies(X)
    y = dataset['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Model Performance:")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    return model, X.columns


# Initialize model
model_data = train_model()
if model_data[0] is None:
    st.stop()

model, feature_columns = model_data

# Sidebar for model information
st.sidebar.markdown("## Model Performance")
st.sidebar.markdown("Training completed successfully!")

st.sidebar.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.markdown("This app uses Linear Regression to predict house prices based on various property features.")

# Main input form
with st.form("house_prediction_form"):
    st.markdown("### Enter House Details")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Details")
        sqft_living = st.number_input(
            "Living Area (sq ft)", 
            min_value=500, 
            max_value=15000, 
            value=2000,
            step=50
        )
        bedrooms = st.number_input(
            "Bedrooms", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        bathrooms = st.selectbox(
            "Bathrooms", 
            options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            index=2
        )
        floors = st.selectbox(
            "Floors", 
            options=[1, 2, 3, 4],
            index=0
        )
    
    with col2:
        st.markdown("#### Property Features")
        sqft_lot = st.number_input(
            "Lot Size (sq ft)", 
            min_value=1000, 
            max_value=50000, 
            value=7500,
            step=100
        )
        grade = st.slider(
            "Grade (Quality)", 
            min_value=1, 
            max_value=13, 
            value=7,
            help="Overall grade given to the housing unit, based on King County grading system"
        )
        condition = st.slider(
            "Condition", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Condition of the house (1=Poor, 5=Excellent)"
        )
        view = st.slider(
            "View Rating", 
            min_value=0, 
            max_value=4, 
            value=0,
            help="An index from 0 to 4 of how good the view of the property was"
        )
    
    with col3:
        st.markdown("#### Additional Info")
        yr_built = st.number_input(
            "Year Built", 
            min_value=1900, 
            max_value=2025, 
            value=1995
        )
        waterfront = st.selectbox(
            "Waterfront", 
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0
        )
        city = st.selectbox(
            "City", 
            options=["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad"]
        )
        garage = st.number_input(
            "Garage Spaces", 
            min_value=0, 
            max_value=5, 
            value=1
        )
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict House Price")

# Prediction logic
if submitted:
    try:
        # Create input dataframe with default values
        input_data = pd.DataFrame({
            'id': [1],
            'date_sold': ['2024-01-01'],
            'city': [city],
            'neighborhood': ['Central'],
            'zipcode': [400001],
            'lat': [19.0760],
            'long': [72.8777],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'grade': [grade],
            'sqft_above': [sqft_living * 0.8],  # Approximate
            'sqft_basement': [sqft_living * 0.2],  # Approximate
            'yr_built': [yr_built],
            'yr_renovated': [0],
            'garage': [garage],
            'parking': [garage + 1],
            'hoa_monthly': [0],
            'price_per_sqft': [0]  # Will be calculated after prediction
        })
        
        # Apply same preprocessing as training data
        input_encoded = pd.get_dummies(input_data)
        
        # Align with training features
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        
        # Display prediction in a nice format
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.markdown(f"## Predicted House Price")
        st.markdown(f"# ₹{prediction:,.0f}")
        st.markdown(f"**Price per sq ft:** ₹{prediction/sqft_living:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Living Area", f"{sqft_living:,} sq ft")
        with col2:
            st.metric("Bedrooms", bedrooms)
        with col3:
            st.metric("Grade", f"{grade}/13")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all fields are filled correctly and the model is properly trained.")

# Additional features section
st.markdown("---")
with st.expander("ℹ️ Model Information & Features"):
    st.markdown("""
    ### About the Model
    This house price prediction model uses **Linear Regression** trained on a dataset of 8,000 house sales.
    
    ### Key Features Used:
    - **Living Area**: Total living space in square feet
    - **Bedrooms & Bathrooms**: Number of rooms
    - **Grade**: Overall quality rating (1-13 scale)
    - **Condition**: Property condition (1-5 scale)
    - **Location**: City and neighborhood
    - **Lot Size**: Total property size
    - **Age**: Year built
    - **Special Features**: Waterfront, view, garage
    
    ### How to Use:
    1. Fill in the house details in the form above
    2. Click "Predict House Price" to get an estimate
    3. The prediction will show the estimated price and price per square foot
    """)

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit** | House Price Prediction ML Model")
# Enhanced footer with developer information
st.markdown("""
<div class="footer">
    <h3>🚀 Built with ❤️ using Streamlit & Machine Learning</h3>
    <div style="margin: 1.5rem 0;">
        <h4 style="color: #667eea; margin-bottom: 0.5rem;">👨‍💻 Developed By</h4>
        <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🎓 Shivansh Mishra</h3>
            <h4 style="margin: 0; color: white;">🎓 Ravi Gupta</h4>
            <h5 style="margin: 0; color: white;">🎓 Shiwanshu Singh</h5>
            <h6 style="margin: 0; color: white;">🎓 Harshvardhan Sisodiya</h6>
            <h7 style="margin: 0; color: white;">🎓 Dhuru Madhuwal</h7>
            <h8 style="margin: 0; color: white;">🎓 Vishal Patel</h8>
            <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                🏛️ <strong>BBD University</strong><br>
                🎯 B.Tech CSE - Cloud Computing & Machine Learning<br>
                📚 Section 2A | 🚀 Future AI Engineer
            </p>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(102, 126, 234, 0.3);">
        <p>🏠 House Price Prediction ML Model | 🤖 Powered by Artificial Intelligence</p>
        <p>✨ Predicting your dream home's value with precision ✨</p>
        <p style="font-size: 0.9rem; color: #667eea; margin-top: 1rem;">
            📧 Academic Project | 🎓 Machine Learning Portfolio | 💡 Innovation in AI
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with developer information
st.sidebar.markdown("""
<div class="metric-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border: 2px solid rgba(102, 126, 234, 0.3);">
    <h3 style="color: #667eea; text-align: center; margin-bottom: 1rem;">👨‍💻 Developers Team Members</h3>
    <div style="text-align: center;">
        <h4 style="color: #764ba2; margin: 0.5rem 0;">🎓 Shivansh Mishra</h4>
        <h5 style="color: #764ba2; margin: 0.5rem 0;">🎓 Ravi Gupta</h5>
        <h6 style="color: #764ba2; margin: 0.5rem 0;">🎓 Shiwanshu Singh</h6>
        <h7 style="color: #764ba2; margin: 0.5rem 0;">🎓 Harshvardhan Sisodiya</h7>
        <h8 style="color: #764ba2; margin: 0.5rem 0;">🎓 Dhuru Madhuwal</h8>
        <h9 style="color: #764ba2; margin: 0.5rem 0;">🎓 Vishal Patel</h9>
        <p style="margin: 0.3rem 0; font-size: 0.9rem; color: #666;">
            🏛️ <strong>BBD University</strong><br>
            🎯 B.Tech CSE (CC & ML)<br>
            📚 Section 2A<br>
            🚀 AI Enthusiast
        </p>
        <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 5px;">
            <p style="margin: 0; font-size: 0.8rem; color: #667eea;">
                💡 Academic ML Project<br>
                🎯 Future AI Engineer
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# End of the app code