import streamlit as st

st.set_page_config(page_title="Landing Page", page_icon="👋")

st.markdown("Landing Page")
st.title("Data Visualization Using Streamlit 👋")
st.subheader("What is Streamlit?")
st.write("""
- Streamlit is an open-source app framework built specifically for Machine Learning and Data Science projects.
- If you want to learn more, you can check out their website [here](https://streamlit.io/)
""")

st.subheader("My projects using Streamlit")
st.write("""
- Project 1:
    - Title: Visualizing the dataset for the paper `The Media Coverage of the 2020 US Presidential Election Candidates through the Lens of Google's Top Stories`
    - Dataset: [available here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0ZLHOK)
- Project 2:
    - Title: Visualizing my spendings
    - Dataset: Manually documented my spendings
""")
