import streamlit as st
import requests
import time


API_URL = "http://13.61.6.118/predict"

st.title("Fraud Detection System")

st.subheader("Enter Transaction Details")

TransactionAmt = st.number_input("Transaction Amount", value=100.0)
ProductCD = st.selectbox("Product Code", ["W", "C", "R", "H", "S"])
card1 = st.number_input("Card1", value=1111.0)
card4 = st.selectbox("Card Type", ["visa", "mastercard", "discover"])
card6 = st.selectbox("Card Category", ["credit", "debit"])
DeviceType = st.selectbox("Device Type", ["desktop", "mobile"])
P_emaildomain = st.text_input("Purchaser Email Domain", "gmail.com")
R_emaildomain = st.text_input("Recipient Email Domain", "gmail.com")

if st.button("Predict"):

    payload = {
        "TransactionAmt": TransactionAmt,
        "ProductCD": ProductCD,
        "card1": card1,
        "card4": card4,
        "card6": card6,
        "DeviceType": DeviceType,
        "P_emaildomain": P_emaildomain,
        "R_emaildomain": R_emaildomain
    }

    start_time = time.time()

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        latency = (time.time() - start_time) * 1000  # ms

        if response.status_code == 200:
            result = response.json()

            st.success(f"Decision: {result['decision']}")
            st.write(f"Fraud Probability: {result['fraud_probability']:.4f}")
            st.write(f"Model Version: {result['model_version']}")
            st.write(f"Latency: {latency:.2f} ms")

        else:
            st.error(f"API Error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Failed: {e}")
