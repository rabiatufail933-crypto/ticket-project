
import streamlit as st
import pandas as pd
import random

# Page Configuration
st.set_page_config(page_title="Intelligent Support System", page_icon="🎫", layout="wide")

st.title("🎫 Intelligent Customer Support System (SaaS)")
st.markdown("Automated Ticket Classification, Sentiment Analysis & Smart Routing")

# Sidebar for Interactive Dashboard
st.sidebar.header("📊 Admin Dashboard & Metrics")
st.sidebar.metric(label="Total Tickets Processed", value="124")
st.sidebar.metric(label="High Priority Tickets", value="18")
st.sidebar.metric(label="Avg Resolution Time", value="2.4 Hours")

st.sidebar.subheader("Team Routing Status")
st.sidebar.text("🟢 Finance Team: Active")
st.sidebar.text("🟢 Logistics Team: Active")
st.sidebar.text("🟢 Technical Support: Active")

# Main Interface
st.subheader("📝 Submit or Test Customer Ticket")

user_input = st.text_area("Enter Customer Ticket / Query:", "My payment was deducted twice for the invoice, please refund my money immediately!", height=120)

# Categories and Keywords Logic
def analyze_ticket(text):
    text_lower = text.lower()
    
    # Category Classification
    if any(word in text_lower for word in ["pay", "charge", "bill", "invoice", "money", "price", "paise"]):
        category = "Billing and Payment Issues"
        route = "Finance Department"
    elif any(word in text_lower for word in ["ship", "deliver", "track", "order", "courier", "address", "parcel"]):
        category = "Shipping and Delivery Issues"
        route = "Logistics Department"
    elif any(word in text_lower for word in ["refund", "return", "cancel", "back", "exchange"]):
        category = "Refunds and Returns"
        route = "Accounts / Returns Team"
    elif any(word in text_lower for word in ["suggest", "good", "bad", "feedback", "improve", "idea"]):
        category = "Customer Feedback and Suggestions"
        route = "Product Management"
    else:
        category = "General Support Issues"
        route = "Customer Care Team"
        
    # Sentiment Analysis
    if any(word in text_lower for word in ["urgent", "bad", "worst", "terrible", "angry", "fake", "immediately", "fraud", "scam"]):
        sentiment = "Negative 😡"
        priority = "🔴 High Priority"
    elif any(word in text_lower for word in ["thanks", "good", "great", "nice", "love", "happy"]):
        sentiment = "Positive 😊"
        priority = "🟢 Low Priority"
    else:
        sentiment = "Neutral 😐"
        priority = "🟡 Medium Priority"
        
    return category, sentiment, priority, route

if st.button("Analyze & Route Ticket", type="primary"):
    if user_input.strip() != "":
        category, sentiment, priority, route = analyze_ticket(user_input)
        
        st.success("Analysis Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("**Category**")
            st.write(category)
            
        with col2:
            st.info("**Sentiment**")
            st.write(sentiment)
            
        with col3:
            st.info("**Priority Level**")
            st.write(priority)
            
        with col4:
            st.info("**Automated Route**")
            st.write(f"➡️ {route}")
            
    else:
        st.warning("Please enter a ticket text first.")
