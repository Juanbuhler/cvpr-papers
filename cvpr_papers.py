import numpy as np
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.events import Tap
from bokeh.models import PanTool, WheelZoomTool
from bokeh.palettes import Paired12, Spectral11, Set1, Set3, Category20, Accent
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


st.set_page_config(layout='wide')
spectral = np.hstack([Category20[20] + Set1[9] + Set3[9] + Accent[8] + Paired12 + Spectral11] * 100)


# Read data from CSV into a pandas DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv("cvpr_data.csv")
    sentence_embeddings = np.load("cvpr_papers_points.npy")
    return df, sentence_embeddings

@st.cache_resource
def get_models():
    sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    nn_model = NearestNeighbors(n_neighbors=num_results, algorithm='auto', metric='cosine')
    nn_model.fit(sentence_embeddings)
    return sentence_model, nn_model

with st.sidebar.popover("About"):
    st.title("A visualization of CVPR papers")
    st.write("Abstracts from all the papers from CVPR 2021 to 2025 were embedded using "
             "https://huggingface.co/sentence-transformers/all-mpnet-base-v2 .")
    st.write("K-means clusters were created at different granularity. For each cluster, the set of paper titles "
             "were passed to a language model to obtain an overarching theme.")
    st.write("Click on a point to read the corresponding abstract and access the paper's page on the CVPR website.")
    st.write("Use the search bar to perform natural language search--queries are embedded using the same model and the"
             " N closest neighbors are highlighted.")
    st.write("Use the controls on the sidebar to color the points according to cluster or year, and to control other "
             "search and display attributes as described.")
    st.markdown("---")
    st.write("Find me at https://www.linkedin.com/in/juanbuhler/")

df, sentence_embeddings = load_data()
projection_method = st.sidebar.selectbox("Projection Method", ["t-SNE", "UMAP"])
point_size = st.sidebar.slider("Point Size", min_value=1, max_value=10, value=3)
st.sidebar.markdown("---")
st.sidebar.write("Select the number of clusters into which to split the data. Each cluster will have an overarching"
                "theme computed from the list of its paper titles.")
num_clusters = st.sidebar.slider("Number of clusters", min_value=20, max_value=100, step=10, value=100)
st.sidebar.write("Then, click on different points on the plot to see what paper they correspond to, read the abstract,"
                 "and go to the CVPR 2024 Open Access paper page")
st.sidebar.markdown("---")
color_by_year = st.sidebar.checkbox("Color by year")
show_year = st.sidebar.selectbox("Show Year", options=["All", "2021", "2022", "2023", "2024", "2025"])
st.sidebar.markdown("---")
num_results = st.sidebar.slider("Number of search results to show", min_value=5, max_value=100, step=1, value=100)

themes_file = f"cluster_themes_{num_clusters}"

with open(themes_file, 'rb') as f:
    themes, cluster_indices = pickle.load(f)

themes_list = [themes[i] for i in cluster_indices]
# Streamlit app layout

title = "CVPR Papers, 2021-2025"
if show_year != "All":
    title = f"CVPR Papers, {show_year}"
st.title(title)
theme = st.selectbox("Select a theme. Points corresponding to the theme selected will be rendered bigger on the graph.", options=['--'] + list(set(themes_list)))
palette = spectral

year_to_id = {2021: 0, 2022: 1, 2023: 2, 2024: 4, 2025: 5}

if color_by_year:
    color = [palette[year_to_id[y]] for y in df["Year"]]
else:
    color = [palette[i] for i in cluster_indices]

size = [point_size] * len(color)

alpha = [.7] * len(color)

query = st.text_input("Or a query for Natural Language Search. Results shown as bigger white points. Clear this box to go back to selecting themes.")

indices = []
if query:
    sentence_model, nn_model = get_models()
    embedded_query = sentence_model.encode([query])[0]

    distances, indices = nn_model.kneighbors(embedded_query.reshape(1, -1), n_neighbors=num_results)
    indices = indices[0]
    # Highlight search results
    alpha = [0.5] * len(color)
    for i in indices:
        if not color_by_year:
            color[i] = "#FFFFFF"
        size[i] = point_size * 3
        alpha[i] = 1
else:
    for i, s in enumerate(size):
        if themes_list[i] == theme:
            size[i] = s * 3

# Create a ColumnDataSource
df['Color'] = color
df['Size'] = size
df['Alpha'] = alpha
df['Theme'] = themes_list

if show_year != "All":
    df = df.loc[df["Year"] == int(show_year)]

source = ColumnDataSource(df)

xarg = "Xtsne"
yarg = "Ytsne"

if projection_method == "UMAP":
    xarg = "Xumap"
    yarg = "Yumap"

# Create a Bokeh plot with dark grey background and black points
p = figure(tools="tap,pan,wheel_zoom,reset", tooltips="@Title", width=700, height=700,
           background_fill_color='#333333', border_fill_color='#333333')


if color_by_year:
    p.scatter(xarg, yarg, source=source, size="Size", color="Color", alpha="Alpha",
              nonselection_fill_alpha=0.75, selection_fill_color='#FFFFFF', legend_field="Year")
else:
    p.scatter(xarg, yarg, source=source, size="Size", color="Color", alpha="Alpha",
              nonselection_fill_alpha=0.75, selection_fill_color='#FFFFFF')
# Customize grid lines
p.xgrid.grid_line_color = '#222222'
p.ygrid.grid_line_color = '#222222'

# JavaScript code to display text and URL on tap
callback = CustomJS(args=dict(source=source), code="""
    const indices = source.selected.indices;
    if (indices.length > 0) {
        const index = indices[0];
        const data = source.data;
        const theme = data['Theme'][index];
        const title = data['Title'][index];
        const abstract = data['Abstract'][index];
        const url = data['URL'][index];
        document.getElementById('theme').innerText = theme;
        document.getElementById('title').innerText = title;
        document.getElementById('abstract').innerText = abstract;
        document.getElementById('url').href = url;
        document.getElementById('url').innerText = url;
    }
""")

p.js_on_event(Tap, callback)

cols = st.columns(2)

cols[0].bokeh_chart(p, use_container_width=True)

cols[0].markdown("<p>Theme:</p>", unsafe_allow_html=True)
cols[0].markdown("<h3>Theme</h3>", unsafe_allow_html=True)
cols[1].markdown("<p id='title'> </p>", unsafe_allow_html=True)
cols[1].markdown("<p id='abstract'>Click on a point to see the details here.</p>", unsafe_allow_html=True)
cols[1].markdown("<a id='url' href='#' target='_blank'></a>", unsafe_allow_html=True)
