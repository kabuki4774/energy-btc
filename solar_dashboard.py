import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt

st.set_page_config(page_title="Solar Bitcoin Mining Dashboard", layout="wide")

# --- Sidebar: Unit economics and project parameters ---
st.sidebar.header("Unit Economics & Assumptions")
solar_cost = st.sidebar.number_input("Solar installation cost ($/W)", value=1.10, step=0.05)
asic_cost = st.sidebar.number_input("ASIC cost ($/TH)", value=75.0, step=5.0)
panel_eff = st.sidebar.slider("Panel efficiency (ηₚ)", 0.10, 0.30, 0.18)
system_derate = st.sidebar.slider("System derate (ηₛ)", 0.70, 1.00, 0.85)
battery_eff = st.sidebar.slider("Battery round-trip eff. (ηᵦ)", 0.80, 1.00, 0.90)
DOD = st.sidebar.slider("Depth of discharge (DOD)", 0.50, 1.00, 0.80)
H_ps = st.sidebar.number_input("Peak-sun hours/day (H_ps)", value=4.0)
G = st.sidebar.number_input("Irradiance G (W/m²)", value=1000.0)
OandM_cost = st.sidebar.number_input("O&M cost ($/kW-year)", value=22.0)
bitcoin_price = st.sidebar.number_input("Bitcoin price ($/BTC)", value=93443.0)
block_reward0 = st.sidebar.number_input("Initial block reward (BTC)", value=6.25)
halving_period = st.sidebar.number_input("Halving period (years)", value=4)
H_net0 = st.sidebar.number_input("Network hashrate (EH/s)", value=854.1) * 1e18
energy_per_TH = st.sidebar.number_input("Energy per TH (J/TH)", value=18.0)

st.sidebar.header("Project & Sensitivity")
area_acres = st.sidebar.number_input("Site area (acres)", value=5.0)
lifetime_years = st.sidebar.number_input("Lifetime (years)", value=10, step=1)
growth_rate = st.sidebar.slider("Hashrate growth (%/yr)", 0.0, 50.0, 10.0) / 100.0
refresh_cycle = st.sidebar.number_input("Hardware refresh cycle (yrs)", value=4, step=1)

# --- Helper functions ---
def block_reward(year):
    return block_reward0 / (2 ** (year // halving_period))

def net_annual_revenue(r, br):
    return 144 * r * br * bitcoin_price * 365 / 1e6  # million $

# --- Static metrics ---
m2_per_acre = 4046.86
area_m2 = area_acres * m2_per_acre
PV_capacity_W = G * panel_eff * system_derate * area_m2
PV_capacity_kW = PV_capacity_W / 1000
avg_power_W = PV_capacity_W * H_ps / 24
hash_rate_THs = avg_power_W / energy_per_TH / 1e3
H_you = hash_rate_THs * 1e12
r0 = H_you / H_net0
BTC_per_day0 = 144 * r0 * block_reward0
hit_prob0 = 1 - math.exp(-144 * r0)
solar_capex0 = PV_capacity_W * solar_cost / 1e6
asic_capex0 = hash_rate_THs * asic_cost / 1e3
bos_capex0 = solar_capex0 * 0.10

# --- Time series simulation ---
years = np.arange(0, lifetime_years+1)
data = []
cum_cf = -(solar_capex0 + asic_capex0 + bos_capex0)
for year in years:
    H_net = H_net0 * ((1 + growth_rate) ** year)
    br = block_reward(year)
    r = H_you * 1e12 / H_net
    rev = net_annual_revenue(r, br)
    oandm = PV_capacity_kW * OandM_cost / 1000
    repl = solar_capex0 if year > 0 and year % refresh_cycle == 0 else 0
    net = rev - oandm - repl
    cum_cf += net
    data.append({
        "Year": year,
        "Revenue": rev,
        "O&M": oandm,
        "Replacement": repl,
        "NetAnnual": net,
        "CumulativeCF": cum_cf,
        "HashShare": r,
        "HashRate_PHs": hash_rate_THs/1e3
    })

df = pd.DataFrame(data)

# --- Main UI ---
st.title("Solar Bitcoin Mining Dashboard")
st.markdown(f"**Scenario:** {area_acres} acres @ ${asic_cost}/TH, {lifetime_years}-yr lifetime")

# Key metrics
c1, c2, c3 = st.columns(3)
c1.metric("PV Capacity (MW)", f"{PV_capacity_kW/1000:.2f}")
c1.metric("Site Area (m²)", f"{area_m2:.0f}")
c2.metric("Hash Rate (PH/s)", f"{hash_rate_THs/1e3:.2f}")
c2.metric("Block Reward", f"{block_reward0:.2f} BTC")
c3.metric("Total CapEx ($M)", f"{solar_capex0+asic_capex0+bos_capex0:.2f}")
payback_year = years[df['CumulativeCF'] >= 0].min() if any(df['CumulativeCF'] >= 0) else None
c3.metric("Payback (yrs)", str(payback_year) if payback_year is not None else ">"+str(lifetime_years))

st.markdown("---")

# --- Time-series charts ---
st.subheader("Time-Series Financials & Cashflow")
chart1 = alt.Chart(df).transform_fold(
    ['Revenue', 'O&M', 'Replacement', 'CumulativeCF'], as_=['Metric', 'Value']
).mark_line().encode(
    x='Year:O', y='Value:Q', color='Metric:N'
).interactive()
st.altair_chart(chart1, use_container_width=True)

st.subheader("Hash Share Over Time")
chart2 = alt.Chart(df).mark_line(color='green').encode(
    x='Year:O', y='HashShare:Q'
).interactive()
st.altair_chart(chart2, use_container_width=True)

# --- Equation Insights Charts ---
st.subheader("Equation Relationship Visualizations")
# 1. PV Area vs Load
P_vals = np.linspace(avg_power_W*0.1, avg_power_W*2, 100)
A_vals = P_vals / (G * panel_eff * system_derate)
df_eq1 = pd.DataFrame({'Load_W': P_vals, 'Area_m2': A_vals})
chart_eq1 = alt.Chart(df_eq1).mark_line().encode(
    x='Load_W', y='Area_m2'
).properties(title='PV Area vs Load (W)')
st.altair_chart(chart_eq1, use_container_width=True)

# 2. Night Energy vs Load
E_vals = P_vals * (24 - H_ps)
df_eq2 = pd.DataFrame({'Load_W': P_vals, 'Night_E_Wh': E_vals})
chart_eq2 = alt.Chart(df_eq2).mark_line(color='orange').encode(
    x='Load_W', y='Night_E_Wh'
).properties(title='Night Energy vs Load')
st.altair_chart(chart_eq2, use_container_width=True)

# 3. Share vs Deployed Hashrate
HR_vals = np.linspace(H_net0*0.01, H_net0*0.5, 100)
r_vals = HR_vals / H_net0
df_eq3 = pd.DataFrame({'H_you_PHs': HR_vals/1e12, 'r': r_vals})
chart_eq3 = alt.Chart(df_eq3).mark_line(color='purple').encode(
    x='H_you_PHs', y='r'
).properties(title='Hash Share vs Deployed PH/s')
st.altair_chart(chart_eq3, use_container_width=True)

# 4. BTC/day vs Share
df_eq4 = pd.DataFrame({'r': r_vals, 'BTC_per_day': 144 * r_vals * block_reward0})
chart_eq4 = alt.Chart(df_eq4).mark_line(color='blue').encode(
    x='r', y='BTC_per_day'
).properties(title='BTC/day vs Hash Share')
st.altair_chart(chart_eq4, use_container_width=True)

# 5. Block Prob vs Share
df_eq5 = pd.DataFrame({'r': r_vals, 'P_block': 1 - np.exp(-144 * r_vals)})
chart_eq5 = alt.Chart(df_eq5).mark_line(color='red').encode(
    x='r', y='P_block'
).properties(title='Daily Block Probability vs Hash Share')
st.altair_chart(chart_eq5, use_container_width=True)

# --- Sensitivity Heatmap ---
st.subheader("Sensitivity Heatmap: Payback by Acres & ASIC Cost")
heat_data = []
sens_acres = np.arange(1, 21)
sens_cost = np.arange(25, 176, 25)
for a in sens_acres:
    for cost in sens_cost:
        area_m2_h = a * m2_per_acre
        pv_w = G * panel_eff * system_derate * area_m2_h
        avg = pv_w * H_ps / 24
        hr = avg / energy_per_TH / 1e3
        solar_c = pv_w * solar_cost / 1e6
        asic_c_h = hr * cost / 1e3
        bos_c_h = solar_c * 0.10
        capex_h = solar_c + asic_c_h + bos_c_h
        net_h = net_annual_revenue(hr * 1e12 / H_net0, block_reward0) - (pv_w/1000 * OandM_cost / 1000)
        pay_h = capex_h / net_h if net_h > 0 else None
        heat_data.append({'Acres': a, 'ASICCost': cost, 'Payback': pay_h})
heat_df = pd.DataFrame(heat_data)
heatmap = alt.Chart(heat_df).mark_rect().encode(
    x='Acres:O', y='ASICCost:O', color='Payback:Q'
).interactive()
st.altair_chart(heatmap, use_container_width=True)

# --- Backend Equations ---
st.markdown("---")
st.subheader("Backend Equations Used")
st.code(
'''A = P / (G * ηₚ * ηₛ)
E_n = P * (24 - H_ps)
C_batt = E_n / DOD
r = H_you / H_net
BTC_per_day = 144 * r * block_reward
P(block) = 1 - exp(-144 * r)
PV_capacity = G * ηₚ * ηₛ * Area
Avg_power = PV_capacity * H_ps / 24
Hash_rate = Avg_power / Energy_per_hash
Solar_CapEx = PV_capacity * solar_cost
ASIC_CapEx = Hash_rate * asic_cost
BOS_CapEx = 0.1 * Solar_CapEx
Total_CapEx = Solar_CapEx + ASIC_CapEx + BOS_CapEx
Annual_revenue = BTC_per_day * 365 * bitcoin_price
Net_annual = Annual_revenue - (PV_capacity/1000 * OandM_cost)
Payback_years = Total_CapEx / Net_annual
''', language='text')
st.markdown("Baseline PDF reference: citeturn0file0")