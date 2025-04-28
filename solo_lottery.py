import streamlit as st
import numpy as np
import pandas as pd

# --------------------
# Constants & Defaults
# --------------------
BLOCK_REWARD = 6.25            # BTC per block
JOULES_PER_KWH = 3.6e6         # Joules in 1 kWh
HASH_FACTOR = 2**32            # Difficulty to hashes factor
POWERBALL_ODDS = 292_201_338   # 1 in 292 million chance for Powerball jackpot

DEFAULTS = {
    "hash_rate_th": 1.0,            # TH/s
    "energy_per_thash": 15.0,       # J/Th
    "electricity_price": 0.10,      # $/kWh
    "network_difficulty": 1.23e14,  # Network difficulty
    "bitcoin_price": 30000.0,       # $ per BTC
    "ticket_size_th": 1e6,          # TH per ticket (1e6 TH = 1e18 hashes)
    "ticket_price": 1.0,            # $ per ticket
    "rigs_count": 1,                # Number of rigs
    "hardware_cost_per_rig": 1000.0,# $ per rig
    "tickets_sold_input": 1000      # Tickets sold per day input
}

# --------------------
# Main App
# --------------------
def main():
    st.set_page_config(page_title="Bitcoin Lottery Dashboard", layout="wide")
    st.title("🎰 Bitcoin Lottery & Solo-Mining Economics Dashboard")
    st.markdown(
        "A turnkey dashboard for evaluating your solo-mining 'lottery machine' and its business model. "
        "Adjust parameters in the sidebar and watch each tab update in real time."
    )

    # Sidebar Inputs
    st.sidebar.header("🔧 Mining Parameters")
    hash_rate_th = st.sidebar.number_input(
        "Hash rate (TH/s)", value=DEFAULTS["hash_rate_th"], format="%.2f"
    )
    energy_per_thash = st.sidebar.number_input(
        "Energy per THash (J/Th)", value=DEFAULTS["energy_per_thash"], format="%.2f"
    )
    electricity_price = st.sidebar.number_input(
        "Electricity price ($/kWh)", value=DEFAULTS["electricity_price"], format="%.2f"
    )
    network_difficulty = st.sidebar.number_input(
        "Network difficulty", value=DEFAULTS["network_difficulty"], format="%.2e"
    )
    bitcoin_price = st.sidebar.number_input(
        "Bitcoin price ($/BTC)", value=DEFAULTS["bitcoin_price"], format="%.2f"
    )

    st.sidebar.header("🎟️ Lottery Parameters")
    ticket_size_th = st.sidebar.number_input(
        "Ticket size (TH)", value=DEFAULTS["ticket_size_th"], format="%.0e"
    )
    ticket_price = st.sidebar.number_input(
        "Ticket price ($)", value=DEFAULTS["ticket_price"], format="%.2f"
    )

    st.sidebar.header("🏢 Business Parameters")
    rigs_count = st.sidebar.number_input(
        "Number of rigs", value=DEFAULTS["rigs_count"], min_value=1, step=1
    )
    hardware_cost_per_rig = st.sidebar.number_input(
        "Hardware cost per rig ($)", value=DEFAULTS["hardware_cost_per_rig"], format="%.2f"
    )
    tickets_sold_input = st.sidebar.number_input(
        "Requested tickets sold per day", value=DEFAULTS["tickets_sold_input"], format="%.0f"
    )

    # --------------------
    # Derived Metrics
    # --------------------
    # Convert to base units
    H = hash_rate_th * 1e12                  # hashes/s
    e = energy_per_thash / 1e12              # J per hash

    # Solo-mining economics
    P = H * e                                 # power draw (W)
    cost_per_hash = electricity_price * (e / JOULES_PER_KWH)
    p_block = 1 / (network_difficulty * HASH_FACTOR)
    hashes_per_block = network_difficulty * HASH_FACTOR
    cost_per_block = hashes_per_block * cost_per_hash
    cost_per_BTC = cost_per_block / BLOCK_REWARD

    # Lottery model
    ticket_hashes = ticket_size_th * 1e12
    p_ticket = ticket_hashes * p_block
    energy_cost_ticket = ticket_hashes * cost_per_hash
    prize_cost_ticket = p_ticket * BLOCK_REWARD * bitcoin_price
    total_cost_ticket = energy_cost_ticket + prize_cost_ticket
    margin_ticket = ticket_price - total_cost_ticket
    margin_pct = margin_ticket / ticket_price * 100

    # Relative odds vs Powerball
    p_powerball = 1 / POWERBALL_ODDS
    odds_ratio = p_ticket / p_powerball
    eq_inve = 1 / p_ticket if p_ticket>0 else float('inf')

    # Business viability
    H_total = H * rigs_count
    capacity_tickets = H_total * 86400 / ticket_hashes
    sold_tickets = min(tickets_sold_input, capacity_tickets)
    if tickets_sold_input > capacity_tickets:
        st.sidebar.warning(
            f"Requested tickets ({tickets_sold_input:.0f}) exceed capacity ({capacity_tickets:.0f}); using max capacity instead."
        )
    revenue_day = sold_tickets * ticket_price
    cost_energy_day = sold_tickets * energy_cost_ticket
    cost_prize_day = sold_tickets * prize_cost_ticket
    profit_day = revenue_day - (cost_energy_day + cost_prize_day)
    profit_month = profit_day * 30
    profit_year = profit_month * 12
    capex_total = hardware_cost_per_rig * rigs_count
    payback_years = capex_total / profit_year if profit_year > 0 else float('inf')
    ROI_pct = profit_year / capex_total * 100 if profit_year > 0 else float('-inf')

    # --------------------
    # Tabs
    # --------------------
    tabs = st.tabs(["Solo-Mining", "Lottery Model", "Viability"])

    # --- Tab 1: Solo-Mining ---
    with tabs[0]:
        st.header("Solo-Mining Economics 📊")
        st.markdown(
            "Each hash is a lottery entry with a tiny chance to win the block reward (6.25 BTC). "
            "Below are your per-hash and per-ticket odds, plus a comparison to Powerball."
        )
        c1, c2 = st.columns(2)
        c1.metric("Power draw (W)", f"{P:.2f}")
        c1.metric("Cost per hash ($)", f"{cost_per_hash:.2e}")
        c1.metric("Win chance per hash", f"{p_block:.2e}")
        c2.metric("Win chance per ticket", f"{p_ticket:.2e}")
        c2.metric("Equivalent odds: 1 in", f"{eq_inve:,.0f}")
        c2.metric("Times more likely than Powerball", f"{odds_ratio:.0f}×")

    # --- Tab 2: Lottery Model ---
    with tabs[1]:
        st.header("Lottery Business Model 🎟️")
        st.markdown(
            "Bundle your hash-power into tickets (TH sized). "
            "See ticket probability, cost breakdown, and margins."
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Win prob/ticket", f"{p_ticket:.2e}")
        d1.metric("Energy cost/ticket ($)", f"{energy_cost_ticket:.4f}")
        d2.metric("Prize cost/ticket ($)", f"{prize_cost_ticket:.4f}")
        d2.metric("Total cost/ticket ($)", f"{total_cost_ticket:.4f}")
        d3.metric("Ticket price ($)", f"{ticket_price:.2f}")
        d3.metric("Margin ($)", f"{margin_ticket:.4f}")
        st.metric("Margin (%)", f"{margin_pct:.2f}%")

    # --- Tab 3: Viability ---
    with tabs[2]:
        st.header("Business Viability 💼")
        st.markdown(
            "Simulate full operations: rigs, capex, ticket volume, daily/annual profit & ROI."
        )
        e1, e2, e3 = st.columns(3)
        e1.metric("Total TH/s", f"{hash_rate_th*rigs_count:.2f}")
        e1.metric("Max tickets/day", f"{capacity_tickets:.0f}")
        e2.metric("Tickets sold/day", f"{sold_tickets:.0f}")
        e2.metric("Daily profit ($)", f"{profit_day:,.2f}")
        e3.metric("Annual profit ($)", f"{profit_year:,.0f}")
        e3.metric("Payback (years)", f"{payback_years:.1f}")
        st.metric("Annual ROI (%)", f"{ROI_pct:.1f}%")

        # Breakdown chart
        breakdown = pd.DataFrame({
            "Category": ["Revenue", "Energy Cost", "Prize Cost"],
            "Daily ($)": [revenue_day, cost_energy_day, cost_prize_day]
        }).set_index("Category")
        st.subheader("Daily Revenue vs. Costs")
        st.bar_chart(breakdown)

        # Sensitivity
        st.subheader("Sensitivity Analysis 🔄")
        ep_min, ep_max = st.slider(
            "Electricity price range ($/kWh)", 0.01, 0.50, (0.05, 0.20), step=0.01
        )
        btc_min, btc_max = st.slider(
            "Bitcoin price range ($)", 10000.0, 100000.0, (20000.0, 60000.0), step=1000.0
        )
        ep_vals = np.linspace(ep_min, ep_max, 20)
        btc_vals = np.linspace(btc_min, btc_max, 20)

        df_ep = pd.DataFrame({"Electricity Price": ep_vals})
        df_ep["Margin %"] = df_ep["Electricity Price"].apply(
            lambda x: ((ticket_price - ((ticket_size_th*1e12)*(x*(energy_per_thash/1e12)/JOULES_PER_KWH) + prize_cost_ticket)) / ticket_price * 100)
        )
        df_btc = pd.DataFrame({"BTC Price": btc_vals})
        df_btc["Margin %"] = df_btc["BTC Price"].apply(
            lambda x: ((ticket_price - (energy_cost_ticket + p_ticket*BLOCK_REWARD*x)) / ticket_price * 100)
        )

        s1, s2 = st.columns(2)
        s1.line_chart(df_ep.set_index("Electricity Price"))
        s2.line_chart(df_btc.set_index("BTC Price"))

if __name__ == "__main__":
    main()
