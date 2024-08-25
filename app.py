import streamlit as st
import main
import os

# Load environment variables
main.load_dotenv(override=True)

# Get notes directories from environment variable
notes_directories = main.notes_directories

@st.cache_resource
def initialize_index():
    if os.path.exists(main.PERSIST_DIR):
        return main.load_index()
    else:
        documents = []
        for directory in notes_directories:
            documents.extend(main.load_documents(directory))
        index = main.create_index(documents, verbose=True)
        main.save_index(index)
        return index

st.title("Document Query System")

# Initialize or load the index
if 'index' not in st.session_state:
    st.session_state.index = initialize_index()

# Sidebar for index update
with st.sidebar:
    st.header("Index Management")
    if st.button("Update Index"):
        main.update_index(st.session_state.index, notes_directories)
        st.success("Index updated successfully!")
        # Force reinitialization of the index
        st.cache_resource.clear()
        st.session_state.index = initialize_index()

# Main query interface
st.header("Query Your Documents")

query = st.text_input("Enter your query:")
model_name = st.selectbox(
    "Select the model for querying:",
    ["gemini-1.5-flash", "gpt-4o", "gpt-4o-mini"],
    index=0
)

if st.button("Submit Query"):
    if query:
        with st.spinner("Processing your query..."):
            response = main.query_index(st.session_state.index, query, model_name)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")

# Display current notes directories
st.header("Current Notes Directories")
for directory in notes_directories:
    st.text(directory)
