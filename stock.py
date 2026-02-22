import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(page_title="Stock Level Predictor", page_icon="📦")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 0.5rem solid #1f77b4;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📦 Stock Level Predictor</div>', unsafe_allow_html=True)
st.markdown("Enter your product parameters below to predict optimal stock levels.")

# Load or train model
@st.cache_resource
def load_model():
    # Generate sample data for training
    np.random.seed(42)
    data = {
        'Price': np.random.uniform(10, 100, 1000),
        'Number of products sold': np.random.uniform(0, 500, 1000),
        'Lead times': np.random.uniform(1, 30, 1000),
        'Order quantities': np.random.uniform(10, 200, 1000),
        'Production volumes': np.random.uniform(50, 1000, 1000),
        'Revenue generated': np.random.uniform(1000, 100000, 1000),
        'Availability': np.random.uniform(0.5, 1.0, 1000),
        'Shipping times': np.random.uniform(1, 14, 1000),
        'Manufacturing lead time': np.random.uniform(1, 20, 1000),
        'Manufacturing costs': np.random.uniform(5, 50, 1000),
        'Defect rates': np.random.uniform(0, 0.1, 1000),
        'Costs': np.random.uniform(100, 5000, 1000),
        'Stock levels': np.random.uniform(100, 2000, 1000)
    }
    df = pd.DataFrame(data)

    # Prepare data
    X = df.drop('Stock levels', axis=1)
    y = df['Stock levels']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    return model, scaler, X.columns.tolist()

model, scaler, feature_names = load_model()

# Input form
st.subheader("📝 Enter Product Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    price = st.number_input("Price ($)", min_value=0.0, value=50.0, step=0.1)
    products_sold = st.number_input("Number of products sold", min_value=0, value=250, step=1)
    lead_times = st.number_input("Lead times (days)", min_value=1, value=15, step=1)
    order_quantities = st.number_input("Order quantities", min_value=1, value=100, step=1)

with col2:
    production_volumes = st.number_input("Production volumes", min_value=1, value=500, step=1)
    revenue = st.number_input("Revenue generated ($)", min_value=0.0, value=50000.0, step=100.0)
    availability = st.number_input("Availability (0-1)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    shipping_times = st.number_input("Shipping times (days)", min_value=1, value=7, step=1)

with col3:
    manufacturing_lead_time = st.number_input("Manufacturing lead time (days)", min_value=1, value=10, step=1)
    manufacturing_costs = st.number_input("Manufacturing costs ($)", min_value=0.0, value=25.0, step=0.1)
    defect_rates = st.number_input("Defect rates (0-1)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
    costs = st.number_input("Costs ($)", min_value=0.0, value=2500.0, step=10.0)

# Predict button
if st.button("🔮 Predict Stock Level", type="primary", use_container_width=True):
    # Prepare input data
    input_data = np.array([[
        price, products_sold, lead_times, order_quantities,
        production_volumes, revenue, availability, shipping_times,
        manufacturing_lead_time, manufacturing_costs, defect_rates, costs
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.markdown(f"""
    <div class="prediction-result">
        <h3>📊 Predicted Stock Level</h3>
        <div class="prediction-value">{prediction:.0f} units</div>
        <p>Based on the parameters you entered, the recommended stock level is <strong>{prediction:.0f} units</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Additional insights
    st.subheader("💡 Quick Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Price Impact", f"${price:.2f}", "per unit")

    with col2:
        st.metric("Demand Level", "High" if products_sold > 300 else "Medium" if products_sold > 100 else "Low")

    with col3:
        st.metric("Lead Time", f"{lead_times} days", "processing time")

else:
    st.info("👆 Enter your parameters above and click 'Predict Stock Level' to get recommendations.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Stock Level Prediction Tool")