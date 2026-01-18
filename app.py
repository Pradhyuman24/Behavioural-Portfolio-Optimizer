import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import shutil
import platform

# ==========================================
# FIX: CACHE & DATA CLEANER
# This handles the 'Database Locked' and 'KeyError' issues
# ==========================================
def fix_yfinance_cache():
    """Forces yfinance to clear corrupted cache files."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "py-yfinance")
    if platform.system() == "Windows":
        # Windows specific cache path
        cache_dir = os.path.join(os.environ.get('LOCALAPPDATA'), 'py-yfinance')
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except Exception:
            pass # Ignore errors if files are in use

# Run cleaner on startup
fix_yfinance_cache()

# ==========================================
# PART 2: ML Models on Bias Detection System
# ==========================================
class BiasDetector:
    def __init__(self):
        pass

    def detect_bias(self, user_responses):
        biases = []
        if user_responses['reaction_to_drop'] == 'Sell immediately':
            biases.append('Loss Aversion')
        if user_responses['trading_freq'] == 'Daily' and user_responses['experience'] < 2:
            biases.append('Overconfidence')
        if user_responses['source_of_info'] == 'Social Media/Friends':
            biases.append('Herding Behavior')
        return biases if biases else ['Rational Investor (No major bias detected)']

# ==========================================================
# PART 3: Bias-adjusted Modern Portfolio Theory Implementation
# ==========================================================
class PortfolioOptimizer:
    def __init__(self, tickers, start_date='2020-01-01'):
        self.tickers = tickers
        self.returns = pd.DataFrame() # Default empty
        
        try:
            # Download data
            raw_data = yf.download(tickers, start=start_date, progress=False)
            
            # SAFETY CHECK 1: Is data empty?
            if raw_data.empty:
                st.error("⚠️ Error: No data downloaded. Please check your internet connection or ticker symbols.")
                return

            # SAFETY CHECK 2: Does 'Adj Close' exist?
            # yfinance structure varies, so we check availability
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                self.data = raw_data['Close']
            else:
                # Fallback if structure is flat
                self.data = raw_data

            # Calculate returns (with fill_method=None to avoid FutureWarnings)
            self.returns = self.data.pct_change(fill_method=None).dropna()
            
        except Exception as e:
            st.error(f"⚠️ Data Error: {str(e)}")
            self.returns = pd.DataFrame()

    def get_metrics(self, weights):
        weights = np.array(weights)
        ret = np.sum(self.returns.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe = ret / vol
        return np.array([ret, vol, sharpe])

    def neg_sharpe(self, weights):
        return -self.get_metrics(weights)[2]

    def optimize_portfolio(self, risk_tolerance, biases):
        # SAFETY CHECK 3: If returns are empty, skip optimization
        if self.returns.empty:
            return np.zeros(len(self.tickers))

        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        min_alloc = 0.0
        max_alloc = 1.0

        # NUDGE SYSTEM
        if 'Loss Aversion' in biases:
            max_alloc = 0.30  # Cap risky assets
        
        if 'Overconfidence' in biases:
            max_alloc = 0.50

        bounds = tuple((min_alloc, max_alloc) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        try:
            result = minimize(self.neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x
        except:
            return init_guess

# ==========================================
# PART 5: User Interface & Dashboard
# ==========================================
def main():
    st.set_page_config(page_title="Zetheta Behavioural Optimizer", layout="wide")
    
    st.title("🧠 Behavioural Portfolio Optimizer")
    st.markdown("### AI-Driven Wealth Management by Detecting Psychological Biases")

    st.sidebar.header("Step 1: Investor Profile")
    tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, GOOG, TSLA, SPY")
    tickers = [t.strip() for t in tickers_input.split(',')]

    st.sidebar.subheader("Psychometric Assessment")
    reaction = st.sidebar.selectbox("Market drops 20% in a week. You:", 
                                    ["Buy more", "Hold", "Sell immediately"])
    freq = st.sidebar.selectbox("Trading Frequency:", ["Daily", "Weekly", "Monthly", "Yearly"])
    exp = st.sidebar.slider("Years of Experience:", 0, 20, 1)
    source = st.sidebar.selectbox("Primary Info Source:", ["Financial Reports", "News", "Social Media/Friends"])

    if st.sidebar.button("Generate Optimization"):
        with st.spinner('Analyzing behavioral patterns and market data...'):
            
            # 1. Run Bias Detector
            detector = BiasDetector()
            user_data = {
                'reaction_to_drop': reaction,
                'trading_freq': freq,
                'experience': exp,
                'source_of_info': source
            }
            biases = detector.detect_bias(user_data)

            # 2. Run Optimizer
            optimizer = PortfolioOptimizer(tickers)
            
            # Only proceed if data was successfully downloaded
            if not optimizer.returns.empty:
                optimal_weights = optimizer.optimize_portfolio(risk_tolerance=0.5, biases=biases)
                
                # --- Display Results ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔍 Behavioral Analysis Results")
                    if 'Rational' in biases[0]:
                        st.success(f"**Detected Profile:** {biases[0]}")
                    else:
                        st.error(f"**Detected Biases:** {', '.join(biases)}")
                        st.info("💡 **Nudge:** The system has automatically adjusted your portfolio constraints to mitigate these risks.")

                with col2:
                    st.subheader("📊 Optimized Allocation")
                    allocation_df = pd.DataFrame({'Asset': tickers, 'Weight': optimal_weights})
                    allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: round(x * 100, 2))
                    st.dataframe(allocation_df)

                st.markdown("---")
                st.subheader("Portfolio Composition")
                fig, ax = plt.subplots()
                ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            else:
                st.warning("⚠️ Could not generate portfolio. Please check ticker symbols and internet connection.")
                
            st.caption("System Integration Status: v1.0.0 | Connected to Yahoo Finance API")

if __name__ == "__main__":
    main()