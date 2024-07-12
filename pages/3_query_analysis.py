import streamlit as st
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Query Analysis", page_icon="🚨")
st.markdown("Query Analysis")
st.title('Query Analysis')

###### FUNCTIONS ######

### POLITCIAL QUERIES FUNCTIONS ###
def get_df_list(directory):
    """
    Returns:
        a tuple of (name, dataframe) in the directory
    """
    df_list = []
    for root , dirs, files in os.walk(directory):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            df_list.append((file, pd.read_csv(file_path)))
    return df_list 

@st.cache_data
def plot_dataframe(df_index):
    df = df_list[df_index][1]
    df.drop(columns=[df.columns[0]], inplace=True)
    df.sort_values(by=["count"])
    df = df.head(20)
    df.rename(columns={"raw-group":"Google Real-Time Trends Cluster", "summarized-query":"paraphrased query"}, inplace=True)

    df['date'] = df['csvFilePath'].apply(lambda x: [pd.to_datetime(filepath.split('/')[-1].split('_')[2].split('@')[-1], format='%m-%d-%H') for filepath in list(x.split(','))])
    df.drop(columns=['csvFilePath', 'platformDirPath'], inplace=True)

    df_query_date = df[['paraphrased query', 'date']]
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab20(range(df_query_date.shape[0]))
    color_map = {query: colors[i] for i, query in enumerate(df_query_date['paraphrased query'])}

    for _, row in df_query_date.iterrows():
        ax.scatter(row['date'], [row['paraphrased query']] * len(row['date']), s=10, color=color_map[row['paraphrased query']], label=row['paraphrased query'])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H'))
    ax.invert_yaxis()
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.yticks(range(df_query_date.shape[0]), df_query_date['paraphrased query'])

    for i, label in enumerate(ax.get_yticklabels()):
        if i < 10:
            label.set_fontweight('bold')
    plt.title(f"Timeline of Top 10 Queries Occurrence for {df_list[df_index][0].split('.csv')[0].split('@')[-1]}")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    st.pyplot(fig)
    return df

@st.cache_data
def createWordCloud(df_list):
    # Concatenate all queries into a single string
    text = ' '
    for name, df in df_list:
        print(df)
        current_text = ' '.join(df['summarized-query'])
        text += current_text

    # Create word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud using matplotlib
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    #plt.show()

    return fig

@st.cache_resource
def get_embeddings():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache_data
def create_embeddings(row,target):
    embed = get_embeddings()
    return embed([row[target]])

@st.cache_data
def clustering_plot(df):
    def draw_cluster_plot(n_clusters,df,matrix,target):
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(matrix)

        df['cluster'] = kmeans.labels_

        #use TSNE to visualize high dim. data
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        tsne_results = tsne.fit_transform(matrix)

        # Step 3: Append tsne_1, tsne_2 
        df_tsne = pd.DataFrame(tsne_results, 
                        columns=['tsne_1', 'tsne_2'])

        df_res = pd.concat([df,df_tsne],axis=1)

        # Step 4: Use plotly to visualize it
        fig = px.scatter(df_res, x='tsne_1', y='tsne_2', color='cluster',hover_data= {target})
        fig.update_traces(textposition='top center', textfont=dict(size=6))
        fig.update_layout(title=f'Embeddings of Google Real-Time Trend Phrases', hoverlabel=dict(
                                        font=dict(size=7),
                                        align="left"))
        return df_res, fig
    
    df = pd.read_csv('aggregated_df.csv')
    df['embeddings'] = df.apply(lambda row: create_embeddings(row,'queries'), axis=1)
    matrix = np.vstack(df['embeddings'])
    df_res, fig = draw_cluster_plot(7,df,matrix,'queries')
    df_clusters = df_res.groupby('cluster')['queries'].agg(list).reset_index()
    return df_clusters, fig



### ALL QUERIES FUNCTIONS ###
def get_date_range(directory, collectedFolder):
    fileNames = []
    for root, dirs, files in os.walk(os.path.join(directory,collectedFolder)):
        for file in files:
                if file.endswith('.csv'):
                    fileNames.append(file.split('.csv')[0])
    return sorted(fileNames)

def agg_df(directory, collectedFolder):
    df_list = []
    for root, dirs, files in os.walk(os.path.join(directory,collectedFolder)):
        for file in files:
            if file.endswith('csv'):
                df_temp = pd.read_csv(os.path.join(root, file), encoding='utf-8')
                if file.startswith('gtrends'):
                    df_temp["date"] = pd.to_datetime(file.split('.csv')[0].split('@')[-1],format='%m-%d-%H')
                else:
                    df_temp["date"] = pd.to_datetime(file.split('.csv')[0],format='%m-%d-%H')
                df_list.append(df_temp)
    return pd.concat(df_list) 

def categorize_response(row):
    if row['response'] < 3:
        return 'non-political'
    else:
        return 'political'
    
def summary_statistics(df):
    """
    Param: 
        df: aggregated dataframe with all raw queries
    """
    if st.checkbox("show Google Trends Query Cluster raw data"):
        st.dataframe(df)
    col1, col2 = st.columns([0.5,0.5])
    col1.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
    df_response = df.groupby('response').size().reset_index(name='count')
    if col2.checkbox("show table for barplot"):
        col2.dataframe(df_response)
    st.bar_chart(data=df_response, x="response",y="count")
    num_of_politcal_queries = df_response[df_response["response"]==5]["count"].values[0] + 1
    num_of_non_politcal_queries = df_response[df_response["response"]==0]["count"].values[0]
    st.write(f"{int(round(num_of_non_politcal_queries/(num_of_non_politcal_queries+num_of_politcal_queries),2)*100)}% of the Google real-time trend clusters are non political queries, labeled by GPT")

    df_date = df.copy()
    df_date['category'] = df.apply(categorize_response, axis=1)
    result = df_date.pivot_table(index='date', columns='category', aggfunc='size', fill_value=0).reset_index()
    st.bar_chart(result, x="date", y=["non-political","political"], color=["#DD66E0", "#0000FF"])

###### CALL STREAMLIT FUNCTIONS ######

### SECTION 1 ###
st.subheader("Section 1: Real Time Query Collection Pipleline Explained")

gpt_example_directory = './data/gpt'

st.write("""
How are these Google Real-Time Trends Cluster collected?
- Using the [pytrends](https://pypi.org/project/pytrends/) library, we set the location to ***United States*** and category to ***Top Stories*** and collected the trending query clusters
- There is a 300 query limit (e.g. [Trump, Biden, Debate] considered 3 queries), which is why we presort by Top Stories (rest of the categories are: Business, Entertainment, Health, Sci/Tech, Sports)
- Then we first ask GPT to identify if a query group is related to US politics or not. 
""")
st.write("This is the prompt we used to finetune GPT 3.5")
st.caption("""
You are a helpful assistant trained to classify queries based on their relevance to current United States politics.
Your role is to evaluate the overall relevance of the given list of queries to current United States politics. 
Your task is to analyze the queries comprehensively and assign a single relevance score from 0 to 5 based on the strength of their collective connection to the political climate, events, figures, or policies in the United States today. 
A score of 0 indicates no relation, while a score of 5 signifies a very strong connection.
""")

if st.checkbox("show training/validation data used for finetuning"):
    st.caption('Since GPT 3.5 labeling results were already satifactory, our finetuning purpose was soley to specify the output format. Thus the training dataset size was not large.')
    df_train = pd.read_csv(os.path.join(gpt_example_directory,'train.csv'))
    df_train.rename(columns={"relevance_jo":"manual_labels"},inplace=True)
    st.dataframe(df_train)

st.divider()
st.write("Then, we needed a second finetuned model to summarized the Google Real-Time Trends clusters into a short query that mimics typical user search behavior.")
st.write("This is the caption we used for this second GPT finetuning task:")
st.caption("""
Given a list of keywords and phrases related to recent news events, 
generate a concise search query that encapsulates the main topics of each list. 
The query should be less than five words and accurately reflect the core elements of the events listed. 
The query should be general enough for internet users seeking information on these topics
""")
if st.checkbox("show some examples"):
    df_example = pd.read_csv('./data/topQueries/topQueries@07-04-12.csv')
    df_example.rename(columns={"raw-group":"Google Real-Time Trends Cluster", "summarized-query":"paraphrased query"}, inplace=True)
    st.dataframe(df_example.head(10)[["Google Real-Time Trends Cluster","paraphrased query"]])

### SECTION 2 ###
st.subheader("Section 2: Top Political Queries Every 12 hours")

political_queries_directory = './data/topQueries'

df_list = get_df_list(political_queries_directory)

for i in range(len(df_list)):
    df = plot_dataframe(i)
    if st.checkbox(f"show raw data {i}"):
        st.dataframe(df)

st.write("""
I wasn't so sure about Texas Patrick Abbott Cyclone and Rieckhoff IAVA Independencd Day, so I looked it up
The original Google trends query cluster is: 
- Dan Patrick, Acting governor, Greg Abbott, Lieutenant Governor of Texas, Tropical cyclone
- Iraq and Afghanistan Veterans of America, ex-serviceman, Paul Rieckhoff, Independence Day (United States)

They had reasons to be classified as political queries. 
- https://www.ltgov.texas.gov/2024/07/06/acting-governor-dan-patrick-adds-81-texas-counties-to-hurricane-beryl-disaster-declaration/
- https://www.abc27.com/business/press-releases/cision/20240702PH53051/independent-veterans-of-america-launches-ahead-of-july-4th/

While observing the data visualizaitons, I realized that there was an error in the code as "SCOTUS biden social media" became "SCOTUS social media biden". This is because I
failed to address the problem taht the sliding window constantly changes the first occurance of the query... modified the query so that it is robust against the sliding window. 
""")

### SECTION 3 ###
directory = "./data"

st.subheader("Section 3: What are these political queries, and how accurate are they?")

df_political_queries = agg_df(directory, "queries")
df_cluster, fig = clustering_plot(df_political_queries)

if st.checkbox("show clustering results"):
    st.write(df_cluster[['cluster','queries']])

st.plotly_chart(fig)


date_range = get_date_range(directory, "queries-raw")
st.write(f"Looked at data collected from {date_range[0]} to {date_range[-1]}")

agg_df_info = agg_df(directory, "queries-raw")
df = agg_df(directory, "queries-raw")
summary_statistics(df)
