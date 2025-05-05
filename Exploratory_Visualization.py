import streamlit as st
from pandas.api.types import is_datetime64_ns_dtype
import warnings
import polars as pl
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
st.set_page_config(layout="wide", page_title="Thomas Wynn - Data Visualization", page_icon='😴') 
warnings.filterwarnings("ignore")
st.sidebar.header("Exploratory Data Visualization")

##########################################################   GET DATA ######################################################### ######
# one_series_path = "/Users/thomaswynn/Desktop/sleep_kaggle/one_series.csv"
# series_events_path = "/Users/thomaswynn/Desktop/sleep_kaggle/one_series_events.csv"
# full_events_path = "/Users/thomaswynn/Desktop/sleep_kaggle/child-mind-institute-detect-sleep-states/train_events.csv"
# full_series_path = "/Users/thomaswynn/Desktop/sleep_kaggle/resampled_train_series.csv"
# fourier_series_path = "/Users/thomaswynn/Desktop/sleep_kaggle/fourier_series.csv"
# fourier_events_path = "/Users/thomaswynn/Desktop/sleep_kaggle/fourier_labels.csv"

one_series_path = "one_series.csv"
series_events_path = "one_series_events.csv"
full_events_path = "child-mind-institute-detect-sleep-states/train_events.csv"
full_series_path = "resampled_train_series.csv"
fourier_series_path = "fourier_series.csv"
fourier_events_path = "fourier_labels.csv"

#@st.cache_data
def get_data(one_series_path, series_events_path, full_events_path, full_series_path) : 
    data = pd.read_csv(one_series_path)
    series_events = pd.read_csv(series_events_path)
    full_train_events = pd.read_csv(full_events_path)
    full_train_series = pd.read_csv(full_series_path)
    fourier_series = pd.read_csv(fourier_series_path).iloc[1:200000, :].iloc[::10, :]
    fourier_events = pd.read_csv(fourier_events_path).iloc[ : 22 , :]

    data.index = pd.to_datetime(data['timestamp'])
    series_events.index = pd.to_datetime(series_events['timestamp'])
    # train_events.index = pd.to_datetime(train_events['timestamp'])
    # train_series.index = pd.to_datetime(train_series['timestamp'])


    wake_steps = series_events[series_events.event == 'wakeup'].timestamp
    onset_steps = series_events[series_events.event == 'onset'].timestamp
    
    wake_steps_df = wake_steps.to_frame(name='timestamp')
    onset_steps_df = onset_steps.to_frame(name='timestamp')

    #Resample to make faster
    enmo_df = data.enmo.resample("120S").mean().to_frame(name = 'enmo')
    anglez_df = data.anglez.resample("120S").mean().to_frame(name = 'anglez')

    awake_enmo = data.awake.resample("120S").mean()
    awake_anglez = data.awake.resample("120S").mean()

    awake_enmo = awake_enmo.reset_index()
    awake_anglez = awake_anglez.reset_index()

    awake_enmo = awake_enmo.drop('timestamp', axis = 1)
    awake_anglez = awake_anglez.drop('timestamp', axis = 1)

    enmo_df = pd.concat((enmo_df.reset_index(), awake_enmo), axis = 1)
    anglez_df = pd.concat((anglez_df.reset_index(), awake_anglez), axis = 1)
    
    return data, enmo_df, anglez_df, onset_steps_df, wake_steps_df, full_train_events, full_train_series, fourier_series, fourier_events

data, enmo_df, anglez_df, onset_steps_df, wake_steps_df, full_train_events, full_train_series, fourier_series, fourier_events = get_data(one_series_path, series_events_path, full_events_path, full_series_path)


def makezero(value) : 
    if value < 1:
        return 0
    else : return 1
        
enmo_df.awake = enmo_df.awake.apply(lambda val : makezero(val))
anglez_df.awake = anglez_df.awake.apply(lambda val : makezero(val))

########################################################  ALTAIR PLOT ############################################################# 

alt.data_transformers.disable_max_rows()
interval = alt.selection_interval(encodings=['x'])

base = alt.Chart(enmo_df.reset_index()).mark_line().encode(
    x = 'timestamp:T',
    y = 'enmo:Q',
    color = 'awake:N'
).properties (
    width = 1400,
    height = 250
)

base2 = alt.Chart(anglez_df.reset_index()).mark_line().encode(
    x = 'timestamp:T',
    y = 'anglez:Q',
    color = 'awake:N'
).properties (
    width = 1600,
    height = 250
)

rule1 = alt.Chart(wake_steps_df).mark_rule(
    color="green",
    strokeWidth=3,
    strokeDash=[20, 1]
).encode(
    x="timestamp:T",
)

rule2 = alt.Chart(onset_steps_df).mark_rule(
    color="purple",
    strokeWidth=3,
    strokeDash=[20, 1]
).encode(
    x="timestamp:T",
)


top = base + rule1 + rule2
top2 = base2 + rule1 + rule2

chart = top.encode(
    x=alt.X('timestamp:T', scale=alt.Scale(domain=interval))
)

chart2 = top2.encode(
    x=alt.X('timestamp:T', scale=alt.Scale(domain=interval))
)

view = base.add_selection(
    interval
).properties (
    height = 50,
    width = 1600
)
my_altair_chart = (chart2 & chart & view).configure_axis(grid = False,   labelFontSize=20,
    titleFontSize=20)


#############################################################  PLOTLY PLOT ################################################################# 

import plotly.express as px
dt_transforms = [
   pl.col('timestamp').str.to_datetime(), 
   (pl.col('timestamp').str.to_datetime().dt.date()).alias('date'), 
   pl.col('timestamp').str.to_datetime().dt.time().alias('time')
]
train_events = (
   pl.scan_csv(full_events_path)
   .with_columns((dt_transforms))
   .collect()
   .to_pandas()
)

my_colors = ["#f79256", "#fbd1a2", "#7dcfb6", "#00b2ca"]
train_events["time"] = train_events.timestamp.dt.time
 
df = train_events.copy()

df.time = pd.to_datetime(df.time, format='%H:%M:%S')
df.set_index(['time'],inplace=True)
df.event = df.event.astype(str)

df.event = df.event.astype(str)

def change_event_numeric(event):
    if event == "onset":
        return 1
    if event == "wakeup":
        return 2
    
df.event = df.event.apply(change_event_numeric)

def jitter(values,j= 0.01):
    return values + np.random.normal(j,0.05,values.shape)


# Create a scatter plot
fig = px.scatter(
    x=df.index,
    y=jitter(df["event"], 0.01),
    color=df["event"],
    size=df["event"] + 4,
    color_continuous_scale = px.colors.qualitative.D3
)

# Set the title and axis labels
fig.update_layout(
    title="Times for onset & wakeup",
    xaxis_title="time",
    yaxis_title="event type",
    width = 1600,
    height = 500,
    yaxis = dict(
        tickmode = 'array',
        tickvals = [1, 2],
        ticktext = ['Onset of Asleep', 'Wake up'],
        tickfont=dict(size=20)
    )
)

fig.update(layout_coloraxis_showscale=False)
fig.update_xaxes(showline=False, linewidth=0, linecolor='white', gridcolor='white', gridwidth= 4)
fig.update_yaxes(showline=False, linewidth=0, linecolor='white', gridcolor='white', gridwidth= 4)

# Rotate the x-axis ticks
fig.update_xaxes(tickangle=45)


########################################################################  APP LAYOUT ##################################################################################

def dataframe_with_selections(df, option):
    df_with_selections = df.copy()
    selected_rows = df_with_selections[df_with_selections['series_id'] == option]
    return selected_rows
st.image("logo.png", width = 500)
st.markdown(f"### This data is from a public Kaggle compeition called [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)")
st.markdown(" \"Child Mind Institute (CMI) transforms the lives of children and families struggling with mental health and learning disorders by giving them the help they need. CMI has become the leading independent nonprofit in children’s mental health by providing gold-standard evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing tomorrow’s breakthrough treatments. Your work will improve researchers' ability to analyze accelerometer data for sleep monitoring and enable them to conduct large-scale studies of sleep. Ultimately, the work of this competition could improve awareness and guidance surrounding the importance of sleep. The valuable insights into how environmental factors impact sleep, mood, and behavior can inform the development of personalized interventions and support systems tailored to the unique needs of each child.\"")

st.markdown("--------")

st.markdown(f"""# Problem Definition & Exploratory Data Analysis
* ### Goal : predict sleep/wakeup times from wrist-worn accelerometer time series data. 
#### We have training data with labeled sleep onset/wake-up times, and our solution will be evaluated on hidden test data, containing only accelerometer measurements.""")

st.markdown('-----------')
st.markdown(f"### Each series ID is a unique identifier for a single, contiguous accelerometer measurement (There are almost 300 unique series!)")
option = st.selectbox(
     'Choose Series ID',
     (train_events.series_id.unique()))

selection_series = dataframe_with_selections(full_train_series, str(option))
selection_events = dataframe_with_selections(full_train_events, str(option))
col1 ,col2 = st.columns([.5,.5], gap="small")
with col1 : 
    st.markdown(f"##### Accelerometer data for series {option}")
    st.write(selection_series.rename({'Unnamed: 0' : "Original Index"}, axis = 1))#.iloc[: , 1::])
with col2 : 
    st.markdown(f"##### Sleep & Wakeup events for series {option}")
    st.write(selection_events)
st.markdown(f"""### Feature Variables : 
* #### anglez (Angular Position of the watch relative to the body's center axis)
* #### enmo (Euclidean Norm Minus One i.e. Angular Acceleration of the watch)""")

st.markdown(f"""### Target Variable : 
* ##### Wakeup  and Sleep Time""")



st.write("----------------------------------------------------------------")
st.markdown(f"""# The relationship between enmo and anglez values and Sleep and Wakeup Times""")
st.markdown(f"""* #### Purple Line : Falls asleep""")
st.markdown(f"""* #### Green Line : Wakes up""")
st.write('#')
st.altair_chart(my_altair_chart, use_container_width= False)

st.write("----------------------------------------------------------------")
st.markdown(f"""# The distribution of Sleep and Wakeup Times """)
st.write("#")
st.plotly_chart(fig)

######################################################################## DENOISING WITH FFT ##################################################################################

st.write("----------------------------------------------------------------")
st.markdown(f"""# Denoising with Fast Fourier Transforms """)
st.write("#")
filter_percentages = [0, 50, 75, 99.7, 99.9, 99.95, 99.99, 99.999]
col1, _ = st.columns([1,3])
with col1:
    thresh = st.slider('Select Power Spectral Density threshold for FFT: ', 0, len(filter_percentages) - 1, 1)

st.markdown(f"### Power Spectral Density Threshold (Percentile): {filter_percentages[thresh]}%")

base1 = alt.Chart(fourier_series.reset_index()).mark_line().encode(
    x = 'timestamp:T',
    y = f'enmo_clean{thresh}:Q',
    
).properties (
    width = 1600,
    height = 300
)
base2 = alt.Chart(fourier_series.reset_index()).mark_line().encode(
    x = 'timestamp:T',
    y = f'anglez_clean{thresh}:Q',
    
).properties (
    width = 1600,
    height = 300
)


st.altair_chart((base1 & base2).configure_axis(grid = False,   labelFontSize=20,
    titleFontSize=20))


#progress_bar = st.sidebar.progress(0)
#`streamlit run /Users/thomaswynn/Desktop/sleep_kaggle/Exploratory_Visualization.py`