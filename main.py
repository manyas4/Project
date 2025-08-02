import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.markdown("""
    <div style='text-align: center; padding: 10px 20px; margin-top: -10px; background: linear-gradient(135deg, #e0f7fa, #ffffff); border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: #0b5394; font-family: "Segoe UI", sans-serif; font-size: 32px; margin-bottom: 8px;'>⚡ Electricity Consumption Trends Prediction</h1>
        <p style='color: #37474F; font-size: 18px; font-family: "Segoe UI", sans-serif; font-style: italic;'>
            Powering Progress: A Visual Analysis of India's Energy Consumption and Potential
        </p>
    </div>
""", unsafe_allow_html=True)

# PDF Export Option
st.markdown("## 🖨 Export as PDF")
show_all = st.checkbox("Show All Sections for PDF Export", value=False)

# CSV 1
df = pd.read_csv("ElecConsumption2.csv")
df.columns = df.columns.str.strip()

st.markdown("<br>", unsafe_allow_html=True)  # line break

states = df["State"].unique()

sorted_states = sorted(states)

default_index = sorted_states.index("Andhra Pradesh")

selected_state = st.selectbox("🔍 Select a State", sorted_states, key="one", index=default_index )

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗂️Category Wise Electricity Consumption", "🔌Installed Capacity and Power Generation", "🌍Renewable Energy : Installed VS Potential Capacity","🔎 Economic Analysis of Electricity Consumption","🔮Future Consumption Prediction"])

#  Tab 1: Donut Chart 
def tab1_content():
    st.subheader(f"🗂️ Category-wise Electricity Consumption Share - {selected_state}")
    st.markdown("<br>", unsafe_allow_html=True)  # line break

    filtered_df = df[df['State'] == selected_state]
    # top consuming sector
    top_sector_row = filtered_df.loc[filtered_df['Percentage Share'].idxmax()]

    top_sector = top_sector_row['Sector']
    top_percentage = top_sector_row['Percentage Share']
    top_consumption = top_sector_row['Consumption (in MU)']

    cols = st.columns(7)

    colors = ['#A3A3A3', '#F4B400', '#DB4437', '#4285F4', '#0F9D58', '#AB47BC', '#FF7043', '#5C6BC0']

    for idx, (i, row) in enumerate(filtered_df.iterrows()):
        with cols[idx % 7]:
            fig = go.Figure(data=[
                go.Pie(
                    labels=[row['Sector'], ''],
                    values=[row['Percentage Share'], 100 - row['Percentage Share']],
                    hole=0.7,
                    marker=dict(colors=[colors[idx % len(colors)], '#E8EAF6']),
                    direction='clockwise',
                    textinfo='none',
                    sort=False,
                    showlegend=False,
                    hoverinfo='text',
                  #  " " empty text for gray part in donut
                    hovertext=[f"{row['Sector']}: {row['Percentage Share']}%<br>Consumption: {row['Consumption (in MU)']} MU", '']
                  )
                  ])

            fig.update_layout(
                title={
                    'text': f"{row['Sector']}: {row['Percentage Share']}%<br>{row['Consumption (in MU)']} MU",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                margin=dict(t=50, b=20, l=0, r=0),
                height=250
            )

            st.plotly_chart(fig, use_container_width=True)
     # description 
    st.markdown(
    f"""
    In **{selected_state}**, the **{top_sector}** sector has the highest electricity consumption share,
    accounting for **{top_percentage}%** of total usage. This corresponds to approximately 
    **{top_consumption} Million Units (MU)** in 2022-23.
    """
)
    
    # Table data for sector-wise classification
    sector_rows = [
    ["🏭 Industrial", "Electricity used by manufacturing, processing, mining, and heavy industry operations. High and continuous demand. Examples: steel, cement, textiles."],
    ["🏢 Commercial", "Business and service establishments — shops, malls, offices, hotels, hospitals. Usage includes lighting, HVAC, and electronic equipment."],
    ["🚂 Railways", "Electricity consumed for train traction and railway infrastructure. Relevant for electrified zones of Indian Railways."],
    ["🚜 Agriculture", "Power for irrigation pumps, agri-machinery, and rural farming needs. Often subsidized and varies seasonally."],
    ["🏠 Domestic", "Household consumption for lighting, fans, TVs, ACs, and daily appliances. Strongly linked with population and lifestyle patterns."],
    ["🏛️ Public Services", "Municipal and government use: street lights, water supply, public buildings, etc. Infrastructure-based demand."],
    ["⚙️ Others / Miscellaneous", "Includes temporary, unclassified, or small-scale usage not fitting standard categories."]
]

    
    df_sectors = pd.DataFrame(sector_rows, columns=["Sector", "Description"])
    with st.expander("🔍 About Sector-wise Electricity Consumption Classification"):
      st.markdown("### ⚡ Sector-wise Electricity Consumption (as per NITI Aayog)")
      st.markdown("""
                  Understanding how different sectors consume electricity helps policymakers, planners, and researchers identify usage patterns, forecast demand, and design targeted energy policies.
         """)
    
      st.table(df_sectors)  # sectors classification table

#Tab 2: Pie Chart 
def tab2_content():
    st.subheader(f"⚙️ Installed Capacity and Power Generation - {selected_state}")
    st.divider()
    
    # CSV 2
    df2 = pd.read_csv("capacityG2.csv")
    df2.columns = df2.columns.str.strip()
    df2 = df2.replace(0, "")

    pie_df = df2[df2['State'] == selected_state]

    pie_df.dropna(subset=["Capacity (in MW)", "Generation (in MU)"], inplace=True)

    # renewable sources
    renewable_sources = {"Solar", "Wind", "Small Hydro", "Biomass"}

    # Top capacity and generation contributors
    top_capacity = pie_df.loc[pie_df["Capacity (in MW)"].idxmax()]
    top_generation = pie_df.loc[pie_df["Generation (in MU)"].idxmax()]

    cap_source = top_capacity["Source"]
    cap_value = top_capacity["Capacity (in MW)"]
    cap_pct = (cap_value / pie_df["Capacity (in MW)"].sum()) * 100

    gen_source = top_generation["Source"]
    gen_value = top_generation["Generation (in MU)"]
    gen_pct = (gen_value / pie_df["Generation (in MU)"].sum()) * 100

    # Classify RE , Non RE
    cap_type = "Renewable" if cap_source in renewable_sources else "Non-Renewable"
    gen_type = "Renewable" if gen_source in renewable_sources else "Non-Renewable"

    # share of renewables
    re_capacity = pie_df[pie_df["Source"].isin(renewable_sources)]["Capacity (in MW)"].sum()
    re_generation = pie_df[pie_df["Source"].isin(renewable_sources)]["Generation (in MU)"].sum()

    total_capacity = pie_df["Capacity (in MW)"].sum()
    total_generation = pie_df["Generation (in MU)"].sum()

    re_capacity_pct = (re_capacity / total_capacity) * 100
    re_generation_pct = (re_generation / total_generation) * 100


    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure(data=[
        go.Pie(
            labels=pie_df["Source"],
            values=pie_df["Capacity (in MW)"],
            textinfo='label+percent',
            name="",  #  This removes "trace 0" from hover
            hovertemplate='%{label}<br>%{value} MW<br>%{percent}',  # What shows on hover
            marker=dict(
            line=dict(color="#fff", width=1  )),
            domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]), # size / positioning of pie chart 


        )
    ])
        fig1.update_layout(margin=dict(t=95, b=60, l=30, r=20),  
         height=400,
         showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<h5 style='text-align:center;'>Installed Capacity (in MW) by Source</h5>",unsafe_allow_html=True)
    with col2:
        fig2 = go.Figure(data=[
        go.Pie(
            labels=pie_df["Source"],
            values=pie_df["Generation (in MU)"],
            textinfo='label+percent',
            marker=dict(line=dict(color="#fff", width=1)),
            name="",  #  removes "trace 0" from hover
            hovertemplate='%{label}<br>%{value} MU<br>%{percent}', 
            domain=dict(x=[0.15, 0.85], y=[0.15, 0.85])

              )        
    ])
        fig2.update_layout(
        margin=dict(t=95, b=60, l=30, r=20),  
        height=400,
        showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<h5 style='text-align:center;'>Electricity Generation (in MU) by Source</h5>",unsafe_allow_html=True)

       
        # description
    st.markdown(
    f"""
    In **{selected_state}**, the source with the highest **installed capacity** is **{cap_source}** 
    ({cap_type}), accounting for **{cap_pct:.1f}%** of the state’s total capacity (**{cap_value} MW**).  
    Meanwhile, **{gen_source}** ({gen_type}) contributes the most to **electricity generation**, 
    producing **{gen_value} MU**, which is **{gen_pct:.1f}%** of the total.

    Overall, **Renewable Energy (RE)** sources make up **{re_capacity_pct:.1f}%** of the state's installed capacity 
    and generate **{re_generation_pct:.1f}%** of its electricity.  
    This comparison highlights the evolving role of renewables in the state's energy landscape.
    """
)   
     # Data for table
    generation_sources = [
    ["🟤 Coal", "Electricity generated by burning coal in thermal power plants. The largest fossil-fuel contributor in India's energy mix."],
    ["🛢️ Oil & Gas", "Power derived from natural gas (combined-cycle plants) and oil-based generators like diesel."],
    ["⚛️ Nuclear", "Low-emission electricity from nuclear reactors. Counted under non-fossil energy."],
    ["🔵 Hydro", "Large hydroelectric projects (>25 MW). Non-fossil, but often reported separately from renewables."],
    ["🔹 Small Hydro", "≤25 MW run-of-river or canal-based hydropower. Included under renewables."],
    ["🔆 Solar", "Electricity from solar PV or solar thermal plants. A major renewable energy source."],
    ["🌬️ Wind", "Power from wind turbines. A clean, renewable source with good scalability."],
    ["♻️ Bio Power", "Electricity from biomass, bagasse cogeneration, biogas, and waste-to-energy. Reliable and renewable."]
]

     # Convert to DataFrame
    source_df = pd.DataFrame(generation_sources, columns=["Source", "Description"])

    with st.expander("🔍 Power Generation Source Classification (NITI Aayog ICED)"):
     st.markdown("### ⚙️ Power Generation Sources (NITI Aayog Classification)")
     st.table(source_df)

     st.markdown("""
    ---
    ### 🧭 Classification Summary

    - 🪨 **Fossil Fuels**: `Coal`, `Oil & Gas` : Non-renewable, carbon-intensive.
    - 🌱 **Renewables**: `Solar`, `Wind`, `Small Hydro`, `Bio Power` : Counted under India's renewable energy targets.
    - ☢️ **Non-Fossil Energy**: `Nuclear`, `Hydro (Large)`, and all **Renewables** : Used to track clean energy share under national/international goals.

    > 🔍 NITI Aayog reports energy data using these classifications to support tracking of India's clean energy transition.
    """)



# Tab 3: Bar Chart 
def tab3_content():
    st.subheader(f"🌿 Renewable Energy: Installed vs Potential Capacity - {selected_state}")
    st.divider()

    df3 = pd.read_csv("RenewableEnergy.csv")
    df3.columns = df3.columns.str.strip()

    pivot_df = df3.pivot_table(index=['State', 'Source'], columns='Capacity Type', values='Capacity (in MW)').reset_index()
    pivot_df.columns.name = None
    pivot_df.columns = ['State', 'Source', 'Installed', 'Potential']

    state_df = pivot_df[pivot_df['State'] == selected_state].copy()

    source_order = ["Hydro", "Wind", "Solar", "Bio Power", "Small-Hydro"]
    state_df['Source'] = pd.Categorical(state_df['Source'], categories=source_order, ordered=True)
    state_df = state_df.sort_values("Source")

    state_df["Utilization (%)"] = (state_df["Installed"] / state_df["Potential"] * 100).round(2).fillna(0)

    colors = {
        "Hydro Installed": "#1abc9c",
        "Hydro Potential": "#b2dfdb",
        "Wind Installed": "#81d4fa",
        "Wind Potential": "#e1f5fe",
        "Solar Installed": "#ffe082",
        "Solar Potential": "#fff3e0",
        "Bio Power Installed": "#33691e",
        "Bio Power Potential": "#c5e1a5",
        "Small-Hydro Installed": "#4dd0e1",
        "Small-Hydro Potential": "#b2ebf2"
    }

    col1, col2 = st.columns(2)
    with col1:
        show_installed = st.checkbox("Display Installed Capacity", value=True)
    with col2:
        show_potential = st.checkbox("Display Potential Capacity", value=True)

    fig3 = go.Figure()

    if show_installed:
        fig3.add_trace(go.Bar(
            x=state_df["Source"],
            y=state_df["Installed"],
            name="Installed",
            marker_color=[colors[f"{src} Installed"] for src in state_df["Source"]],
            text=state_df["Installed"],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Installed: %{y} MW<br>Utilization: %{customdata:.2f}%',
            customdata=state_df["Utilization (%)"]
        ))

    if show_potential:
        fig3.add_trace(go.Bar(
            x=state_df["Source"],
            y=state_df["Potential"],
            name="Potential",
            marker_color=[colors[f"{src} Potential"] for src in state_df["Source"]],
            text=state_df["Potential"],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Potential: %{y} MW'
        ))

    fig3.update_layout(
        barmode='group',
        title=f"Renewable Energy Installed vs Potential Capacity - {selected_state}",
        xaxis_title="Energy Source",
        yaxis_title="Capacity (in MW)",
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.divider()
    #  top potential source
    top_potential_row = state_df.loc[state_df["Potential"].idxmax()]
    top_potential_source = top_potential_row["Source"]
    top_potential_value = top_potential_row["Potential"]

    # top utilization %
    top_util_row = state_df.loc[state_df["Utilization (%)"].idxmax()]
    top_util_source = top_util_row["Source"]
    top_util_rate = top_util_row["Utilization (%)"]

    # Total installed vs potential capacity
    total_installed = state_df["Installed"].sum()
    total_potential = state_df["Potential"].sum()
    total_util_pct = (total_installed / total_potential * 100).round(2) if total_potential != 0 else 0

    #  summary
    st.markdown(
    f"""
    In **{selected_state}**, the renewable source with the highest potential is **{top_potential_source}**, 
    offering up to **{top_potential_value} MW** of capacity.  

    The **highest utilization** is observed in **{top_util_source}**, where **{top_util_rate}%** of its potential has already been installed.  

    Overall, the state has utilized **{total_util_pct}%** of its total renewable energy potential, with an installed capacity of **{total_installed:.2f} MW** out of **{total_potential} MW**.
    """
 )  
    
    re_sources= [
    ["🔆 Solar", "✅ Yes", "High potential across most Indian states."],
    ["🌬️ Wind", "✅ Yes", "Onshore wind potential mapped by NIWE."],
    ["🔹 Small Hydro", "✅ Yes", "≤25 MW, officially counted under RE targets."],
    ["🔵 Hydro (Large)", "✅ *Yes* (in this section)", "Included in ICED’s RE potential/installed view, even if not always counted in official RE targets."],
    ["♻️ Bio Power", "✅ Yes", "Includes biomass, bagasse cogeneration, waste-to-energy."]
]
    
     # Convert to DataFrame
    re_source_df = pd.DataFrame(re_sources, columns=["Source", "Included in RE?", "Remarks"])

    with st.expander("🌿 Renewable Energy: Installed vs Potential Capacity (NITI Aayog ICED)"):
     st.markdown("""
    ### 🌞 Installed vs Potential Renewable Energy Capacity

    The NITI Aayog India Climate & Energy Dashboard (ICED) includes the following sources in its **Renewable Energy Installed vs Potential Capacity** section:
    """ )

     st.table(
     pd.DataFrame(re_source_df)
     )
     st.markdown("""
    ### 📌 Important Note

    - Although **Large Hydro** (>25 MW) is traditionally **not included** in India's official **renewable energy targets**, it **is included** in the **Installed vs Potential** figures on ICED.
    - This makes the renewable capacity figures more **comprehensive**, but slightly different from policy-specific definitions where only **Small Hydro** is counted under RE.

    """, unsafe_allow_html=True)

#st.markdown("<hr style='border: 1px solid brown;'>", unsafe_allow_html=True)
st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(to right, #1a73e8, #00bcd4);'>", unsafe_allow_html=True)

def tab4_content():

    st.subheader("📉 Per Capita GDP vs Electricity Consumption - INDIA")
    st.divider()

    df4 = pd.read_csv("GDP.csv")
    df4.columns = df4.columns.str.strip()
    df4 = df4.dropna()

    state_abbreviations = {
        'Andaman and Nicobar Islands': 'AN', 'Andhra Pradesh': 'AP', 'Arunachal Pradesh': 'AR',
        'Assam': 'AS', 'Bihar': 'BR', 'Chandigarh': 'CH', 'Chhattisgarh': 'CG', 'Delhi': 'DL',
        'Goa': 'GA', 'Gujarat': 'GJ', 'Haryana': 'HR', 'Himachal Pradesh': 'HP', 'Jharkhand': 'JH',
        'Karnataka': 'KA', 'Kerala': 'KL', 'Madhya Pradesh': 'MP', 'Maharashtra': 'MH', 'Manipur': 'MN',
        'Meghalaya': 'ML', 'Mizoram': 'MZ', 'Nagaland': 'NL', 'Odisha': 'OD', 'Punjab': 'PB',
        'Rajasthan': 'RJ', 'Sikkim': 'SK', 'Tamil Nadu': 'TN', 'Telangana': 'TG', 'Tripura': 'TR',
        'Uttar Pradesh': 'UP', 'Uttarakhand': 'UK', 'West Bengal': 'WB','Puducherry': 'PY'
    }

    df4['Short'] = df4['State'].map(state_abbreviations)

    all_states_list = df4['State'].dropna().unique().tolist()
    all_states_list.sort()

    # Add 'ALL STATE' option on top of list
    state_options = ['ALL STATES'] + all_states_list

    # Multiselect with 'ALL STATES' as default
    selected_states = st.multiselect(

    "🌐 Select States ", options=state_options,  default=['ALL STATES'], key="four")

   # Filter logic
    if 'ALL STATES' in selected_states:
     filtered_df = df4[df4['State'].isin(all_states_list)]
    else:
     filtered_df = df4[df4['State'].isin(selected_states)]

    fig4 = go.Figure()

    # Convert per capita GSDP from crore to ₹ (e.g., 0.029 Cr → ₹29,00,000)
    filtered_df["Per Capita GSDP (Rs)"] = (filtered_df["GSDP per capita (Rs in crore)"] * 1e7).round(0)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
     x=filtered_df['Consumption (in kWh)'],
     y=filtered_df['GSDP Current Price (Rs in crore)'],
     mode='markers+text',
     text=filtered_df['Short'],
     textposition='middle center',
     textfont=dict(size=14, color='black'),
     marker=dict(
        size=filtered_df['Consumption (in kWh)'],
        sizemode='area',
        color=filtered_df['GSDP Current Price (Rs in crore)'],
        colorscale='YlGnBu',
        showscale=True,
        colorbar=dict(title='GSDP (₹ Cr)'),
        line=dict(width=2, color='black')
    ),
     hovertemplate=(
        "<b>%{customdata[0]}</b><br>" +
        "GSDP: ₹%{customdata[1]:,} Cr<br>" +
        "Consumption: %{x} kWh<br>" +
        "Population: %{customdata[2]:,}<br>" +
        "Per Capita GSDP: ₹%{customdata[3]:.3f} Cr"
    ),
     customdata=filtered_df[['State', 'GSDP Current Price (Rs in crore)', 'Population', 'GSDP per capita (Rs in crore)']].values
))

    fig4.update_layout(
            title='Per Capita GDP vs Per Capita Electricity Consumption',
            xaxis_title='Electricity Consumption Per Capita (in kWh)',
            yaxis_title='GSDP at Current Prices (in Rs. Crores)',
            height=600
    )

    fig4.update_xaxes(tickvals=[0,500,1000,1500,2000,2500,3000,3500], range=[0, 3500])

    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown(" 📊 About Chart")
    st.markdown("""
    This bubble chart shows the 2022-23 data for Indian states, comparing **Per Capita Electricity Consumption** and **GSDP**:

    - **X-axis**: Electricity consumption per person (kWh)  
    - **Y-axis**: Total GSDP (₹ crore)  
    - **Bubble size**: Population  
    - **Bubble color**: Per capita GSDP  

    Hover to see full state name, consumption, GSDP, population, and per capita GSDP.  
    This helps identify how economic output relates to electricity use across states.
    """)
    with st.expander("📈 For more info."):
     st.markdown("""
    ### 💡 Understanding Economic Growth & Electricity Use

    This section explores the relationship between a state's **economic development** and its **electricity consumption**.

     **Per Capita GDP (US $)** : A measure of average income and economic output per person. <br> 
     **Per Capita Electricity Consumption (kWh)** : Indicates access to and usage of energy per person. 

    ### 🔍 Why It Matters

    - Higher electricity consumption often reflects **higher industrialization, urbanization**, and **living standards**.
    - Helps identify states with **untapped economic potential** or **underutilized infrastructure**.
    - Useful for **policy planning** — e.g., areas needing energy access improvements or demand management.

    
    """, unsafe_allow_html=True)
     
# TAB 5
def tab5_content():
      df5 = pd.read_csv("new.csv")  

      df5.rename(columns={ "Per_Capita_GDP(US $)": "GDP"}, inplace=True)

      # cleaning year column
      df5['Year'] = df5['Year'].astype(str).str.split('-').str[0].astype(int)

      df5['GDP'] = df5['GDP'].astype(str).str.replace(',', '')
      df5['Population'] = df5['Population'].astype(str).str.replace(',', '')
      df5 = df5.dropna()
      df5['GDP'] = pd.to_numeric(df5['GDP'])
      df5['Population'] = pd.to_numeric(df5['Population'])

      # train model
      X = df5[['Year', 'GDP', 'Population']]
      y = df5['Per_Capita_Consumption(kWh)']

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      model = LinearRegression()
      model.fit(X_train, y_train)
      r2 = r2_score(y_test, model.predict(X_test))

      st.subheader("🔌 Predict Per Capita Electricity Consumption - INDIA")
      st.divider()
      colA , colB = st.columns([0.4,0.6])
      with colA:
         # Year selection based on recent range
        future_year = st.selectbox("Select Prediction Year (for India)", options=list(range(2022, 2031)))

        # GDP  (in US $ per capita)
        future_gdp = st.slider("Enter Projected Per Capita GDP (US $)",
         min_value=1000.0,
         max_value=5000.0,
         value=2358.0,
         step=2.0,format="%.0f")

        # Population (in Crores)
        future_population_crores = st.slider("Enter Projected Population (in Crores)",
        min_value=120.0,
        max_value=200.0,
        value=145.5,
        step=0.1,format="%.1f")
        future_population = future_population_crores * 1e7

        predicted_consumption = model.predict([[future_year, future_gdp, future_population]])[0]
        
        st.markdown("**Last 5 years data for reference :**<br>• GDP (Per Capita GDP in US $)<br>• Population (in Crores)",unsafe_allow_html=True)
        st.dataframe(df5.tail())
      
      with colB:
         # Output
        st.markdown(f"### 📈 Predicted Per Capita Consumption for {future_year}: **{predicted_consumption:.2f} kWh**")
        st.caption(f"Using Input GDP = ${future_gdp:,.0f} and Population = {future_population_crores:.1f} Crores")
       # st.markdown(f"✅ **R² Score (Test Data):** {r2:.4f}")

        fig5 = go.Figure()

        # Historical data
        fig5.add_trace(go.Scatter(
         x=df5["Year"],
        y=df5["Per_Capita_Consumption(kWh)"],
         mode='lines+markers',
        name='Historical Consumption',
        line=dict(color='orange', width=3),
        marker=dict(color='limegreen', size=10, symbol='circle')
))

        # Predicted point
        fig5.add_trace(go.Scatter(
        x=[future_year],
        y=[predicted_consumption],
        mode='markers',
        name='Predicted Consumption',
        marker=dict(color='red', size=12, symbol='x')
))

        fig5.update_layout(
    title="Per Capita Electricity Consumption in India: Historical and Projected",
    xaxis_title="Year",
    yaxis_title="Per Capita Consumption (kWh)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
)

        st.plotly_chart(fig5)
   

# --- Show all sections or use tabs ---
if show_all:
    st.markdown("""
    <script>
        window.addEventListener('load', function() {
            window.print();
        });
    </script>
    """, unsafe_allow_html=True)

    tab1_content()
    tab2_content()
    tab3_content()
    tab4_content()
    tab5_content()
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️Category Wise Electricity Consumption", "🔌Installed Capacity and Power Generation",
     "🌍Renewable Energy : Installed VS Potential Capacity","🔎 Economic Analysis of Electricity Consumption",
     "🔮Future Consumption Prediction"
    ])
    with tab1: tab1_content()
    with tab2: tab2_content()
    with tab3: tab3_content()
    with tab4: tab4_content()
    with tab5: tab5_content()

