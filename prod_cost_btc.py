import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Utility Functions & Equations
# -------------------------
def fetch_btc_price():
    """Fetch current BTC price (USD) from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        price = data["bitcoin"]["usd"]
    except Exception as e:
        st.error("Error fetching BTC price: " + str(e))
        price = None
    return price

def compute_cost_per_btc(hash_rate_EH, efficiency_J_per_TH, elec_cost, block_reward):
    """
    Compute the cost to mine one BTC.
    
    Parameters:
    - hash_rate_EH: Network hash rate in exahashes per second (EH/s)
    - efficiency_J_per_TH: Average energy consumption in Joules per terahash (J/TH)
    - elec_cost: Electricity cost in $/kWh
    - block_reward: BTC rewarded per block (e.g., 3.125 post-halving)
    
    Equations:
    - Convert network hash rate to TH/s: H_TH = hash_rate_EH * 1e6
    - Daily energy (J) = H_TH * efficiency * 86400
    - Convert Joules to kWh: divide by 3.6e6
    - BTC mined per day = block_reward * 144 (approx. 144 blocks/day)
    
    So:
    \[
    \text{Cost per BTC} = \frac{H_{TH} \times E \times 86400}{3.6 \times 10^6 \times (block\_reward \times 144)} \times elec\_cost
    \]
    """
    H_TH = hash_rate_EH * 1e6  # convert EH/s to TH/s
    energy_kWh_per_day = (H_TH * efficiency_J_per_TH * 86400) / 3.6e6
    btc_per_day = block_reward * 144
    cost_per_btc = (energy_kWh_per_day * elec_cost) / btc_per_day
    return cost_per_btc

def simulate_future_costs(
    current_hash_rate_EH,
    current_efficiency,
    elec_cost,
    block_reward,
    hash_rate_growth,  # annual growth rate of hash rate (%)
    efficiency_improvement,  # annual % decrease in J/TH
    years=10
):
    """
    Simulate future mining cost per BTC for a given number of years.
    Assume:
      - Hash rate grows at hash_rate_growth per year.
      - Efficiency improves (i.e. lower J/TH) by efficiency_improvement per year.
      - Electricity cost and block reward remain constant over the simulation horizon.
    """
    years_array = np.arange(0, years + 1)
    costs = []
    hr = current_hash_rate_EH
    eff = current_efficiency
    for t in years_array:
        cost = compute_cost_per_btc(hr, eff, elec_cost, block_reward)
        costs.append(cost)
        # Update parameters for next year:
        hr *= (1 + hash_rate_growth/100.0)
        eff *= (1 - efficiency_improvement/100.0)
    return years_array, np.array(costs)

# -------------------------
# Streamlit Dashboard
# -------------------------
st.set_page_config(page_title="Bitcoin & Energy Investment Thesis Dashboard", layout="wide")
st.title("Bitcoin & Energy Investment Thesis Dashboard")
st.markdown(
    """
    This dashboard integrates Bitcoin’s key on-chain and mining fundamentals with broader energy economics and macro trends.  
    It is structured along the “energy-to-Bitcoin vertical”: starting from fixed supply and halving events, through mining difficulty, hash rate, ASIC efficiency, and electricity cost, which together create a production cost “floor” that supports Bitcoin’s price.  
    
    Use the sidebar to adjust assumptions, view key equations, and see simulated future production costs (a proxy for the price floor).  
    The dashboard also fetches current BTC price data (via CoinGecko) so you can compare market price to production cost.
    """
)

# Sidebar for Adjustable Assumptions
st.sidebar.header("Adjust Assumptions")
st.sidebar.subheader("Current Network Parameters")
current_hash_rate_EH = st.sidebar.number_input(
    "Network Hash Rate (EH/s)", value=878, step=0.01, help="Enter the current network hash rate in exahashes per second (EH/s)."
)
current_efficiency = st.sidebar.number_input(
    "Average ASIC Efficiency (J/TH)", value=21.0, step=1.0, help="Enter the average energy consumption in Joules per terahash (J/TH)."
)
elec_cost = st.sidebar.number_input(
    "Electricity Cost ($/kWh)", value=0.06, step=0.005, help="Enter the electricity cost in USD per kWh."
)
block_reward = st.sidebar.number_input(
    "Block Reward (BTC)", value=3.125, step=0.001, help="Enter the current block reward in BTC (post-halving)."
)

st.sidebar.subheader("Assumptions for Future Projections")
hash_rate_growth = st.sidebar.slider(
    "Annual Hash Rate Growth (%)", min_value=0.0, max_value=50.0, value=20.0, step=1.0,
    help="Estimated annual percentage growth in the network hash rate."
)
efficiency_improvement = st.sidebar.slider(
    "Annual Efficiency Improvement (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5,
    help="Estimated annual percentage decrease in J/TH (improvement in ASIC efficiency)."
)

st.sidebar.markdown("---")
st.sidebar.header("Key Equations")
st.sidebar.latex(r"""
\text{Cost per BTC} = \frac{H_{TH} \times E \times 86400}{3.6\times 10^6 \times (B \times 144)} \times C
""")
st.sidebar.markdown(
    """
    where:  
    - \(H_{TH}\) is the network hash rate in TH/s (1 EH/s = \(10^6\) TH/s)  
    - \(E\) is the average ASIC efficiency (J/TH)  
    - \(C\) is the electricity cost (USD/kWh)  
    - \(B\) is the block reward (BTC)  
    - 86400 is the number of seconds in a day  
    - 144 is the average number of blocks per day  
    """
)

# -------------------------
# Current Metrics Section
# -------------------------
st.header("Current Network Metrics & Cost of Production")
btc_price = fetch_btc_price()
if btc_price:
    st.metric("Current BTC Price (USD)", f"${btc_price:,.2f}")
else:
    st.write("BTC price data not available.")

cost_per_btc = compute_cost_per_btc(current_hash_rate_EH, current_efficiency, elec_cost, block_reward)
st.metric("Estimated Production Cost per BTC (USD)", f"${cost_per_btc:,.2f}")

st.markdown(
    f"""
    Based on the current assumptions:
    - **Network Hash Rate:** {current_hash_rate_EH} EH/s  
    - **ASIC Efficiency:** {current_efficiency} J/TH  
    - **Electricity Cost:** ${elec_cost:.3f}/kWh  
    - **Block Reward:** {block_reward} BTC  

    The calculated production cost is derived as:
    \[
    \text{{Cost per BTC}} = {cost_per_btc:,.2f} \text{{ USD}}
    \]
    This metric represents the network’s “floor” — if BTC falls well below this value for an extended period, miners may be forced to shut down, reducing supply pressure.
    """
)

# -------------------------
# Future Projections Simulation
# -------------------------
st.header("Future Production Cost Simulation")

years_5, costs_5 = simulate_future_costs(
    current_hash_rate_EH, current_efficiency, elec_cost, block_reward,
    hash_rate_growth, efficiency_improvement, years=5
)
years_10, costs_10 = simulate_future_costs(
    current_hash_rate_EH, current_efficiency, elec_cost, block_reward,
    hash_rate_growth, efficiency_improvement, years=10
)

# Create a DataFrame for visualization
df_5 = pd.DataFrame({"Year": years_5, "Production Cost per BTC (USD)": costs_5})
df_10 = pd.DataFrame({"Year": years_10, "Production Cost per BTC (USD)": costs_10})

st.subheader("Production Cost Projection Over Next 5 Years")
st.line_chart(df_5.set_index("Year"))
st.dataframe(df_5)

st.subheader("Production Cost Projection Over Next 10 Years")
st.line_chart(df_10.set_index("Year"))
st.dataframe(df_10)

st.markdown(
    """
    **Simulation Explanation:**  
    - We assume that the network hash rate grows by the selected annual rate.
    - ASIC efficiency improves (i.e. lower J/TH) by the selected percentage each year.
    - Electricity cost and block reward remain fixed over the simulation period.
    
    The simulation computes the production cost per BTC each year using the formula:
    \[
    \text{{Cost per BTC}} = \frac{H_{TH}(t) \times E(t) \times 86400}{3.6\times10^6 \times (B \times 144)} \times C
    \]
    where \(H_{TH}(t)\) and \(E(t)\) evolve based on the annual growth and improvement rates.
    """
)

# -------------------------
# Probabilistic Framework & Investment Thesis
# -------------------------
st.header("Investment Thesis & Probabilistic Framework")
st.markdown(
    """
    Investment thesis integrates the following key components:
    
    1. **Scarcity & Supply Dynamics:**  
       - The total BTC supply is capped at 21 million with diminishing new issuance due to halving events.  
       - With over 93% of BTC mined and the 2024 halving cutting rewards to 3.125 BTC, scarcity is intensifying.
    
    2. **Mining Economics as a Price Floor:**  
       - The network’s production cost (driven by hash rate, ASIC efficiency, and electricity costs) establishes a fundamental floor below which price cannot sustainably fall.  
       - If BTC price falls significantly below this floor, miner capitulation is likely, reducing supply and eventually supporting a price rebound.
    
    3. **Network Security & Hash Rate:**  
       - An ever-increasing hash rate (now nearing record EH/s levels) reinforces network security and implies robust miner confidence.  
       - Difficulty adjusts to maintain consistent block times; rising difficulty means higher production costs over time.
    
    4. **Technological & Energy Trends:**  
       - ASIC efficiency improvements (although maturing) and the steady availability of low-cost power (often sourced from renewables) are key to sustaining mining profitability.  
       - Competition from the AI boom for chips might slightly constrain hardware availability, but miners are actively adapting.
    
    5. **Macroeconomic & Adoption Drivers:**  
       - Growing institutional acceptance, a hedge against inflation, and potential macro pivots (e.g. easing policies) further drive Bitcoin’s value.
    
    **Probabilistic Outcome Scenarios:**  
    Based on this framework, we envision three outcome scenarios for Bitcoin’s price:
    
    | Scenario        | Short-Term (1–2 Years)                                  | Long-Term (2030)                                  |
    |-----------------|---------------------------------------------------------|---------------------------------------------------|
    | **Conservative**| Minor correction toward the production cost floor (e.g. 10–30% dip)  | \$80k – \$120k – modest appreciation driven by supply constraints |
    | **Baseline**    | Cyclical volatility with price staying above mining costs           | \$150k – \$250k – steady growth aligned with network fundamentals |
    | **Optimistic**  | Robust demand with temporary euphoria driving higher multiples         | \$300k – \$500k+ – widespread adoption and “digital gold” dynamics |
    
    These scenarios are underpinned by the interplay of mining economics and broader macro factors. Investors should monitor:
    
      - **Network Hash Rate & Difficulty:** Higher values indicate rising production costs and increased security.
      - **ASIC Efficiency & Electricity Costs:** They determine the baseline cost of production.
      - **Institutional Adoption & Macro Trends:** They influence demand and the eventual market premium above production costs.
    
    **Guidance:**  
    Use the controls above to adjust assumptions and see how production cost evolves. Compare that “floor” to current market prices and consider the probabilistic outcomes when evaluating your investment horizon.
    """
)

st.markdown(
    """
    ---
    
    **Note:** This dashboard is a tool to help visualize and experiment with key components of the energy-to-Bitcoin value chain. The underlying equations and simulation are based on simplified models. While historical trends provide useful anchors, actual market outcomes depend on many factors, including rapid changes in technology, energy prices, macroeconomic policies, and unexpected events.
    
    Please use this dashboard as one of many tools in your investment research.
    """
)
