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

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for index update and conversation management
with st.sidebar:
    st.header("Index Management")
    if st.button("Update Index"):
        main.update_index(st.session_state.index, notes_directories)
        st.success("Index updated successfully!")
        # Force reinitialization of the index
        st.cache_resource.clear()
        st.session_state.index = initialize_index()
    
    st.header("Conversation Management")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.success("Conversation cleared!")

# Main query interface
st.header("Chat with Your Documents")

# Display chat history
for i, (role, message) in enumerate(st.session_state.chat_history):
    with st.chat_message(role):
        st.write(message)

# Query input
query = st.chat_input("Enter your query:")
model_name = st.selectbox(
    "Select the model for querying:",
    ["gemini-1.5-flash", "gpt-4o", "gpt-4o-mini"],
    index=0
)

if query:
    # Add user message to chat history
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    # Prepare context from chat history
    context = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history[-5:]])

    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            response = main.query_index(st.session_state.index, query, model_name, context)
        st.write(response)

    # Add assistant response to chat history
    st.session_state.chat_history.append(("assistant", response))

# Display current notes directories
st.header("Current Notes Directories")
for directory in notes_directories:
    st.text(directory)
