import streamlit as st
import main
import os

# Load environment variables
main.load_dotenv(override=True)

# Initialize session state for notes directories
if 'notes_directories' not in st.session_state:
    st.session_state.notes_directories = main.notes_directories

@st.cache_resource
def initialize_index():
    if os.path.exists(main.PERSIST_DIR):
        return main.load_index()
    else:
        documents = []
        for directory in st.session_state.notes_directories:
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

# Sidebar for index update, conversation management, and notes directories
with st.sidebar:
    st.header("Index Management")
    if st.button("Update Index"):
        main.update_index(st.session_state.index, st.session_state.notes_directories)
        st.success("Index updated successfully!")
        # Force reinitialization of the index
        st.cache_resource.clear()
        st.session_state.index = initialize_index()
    
    st.header("Conversation Management")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.success("Conversation cleared!")

    st.header("Notes Directories")
    # Display and edit current notes directories
    st.text_area("Current Notes Directories (one per line)", 
                 value="\n".join(st.session_state.notes_directories), 
                 key="notes_dirs_input")
    
    if st.button("Update Notes Directories"):
        new_dirs = st.session_state.notes_dirs_input.split("\n")
        new_dirs = [dir.strip() for dir in new_dirs if dir.strip()]
        st.session_state.notes_directories = new_dirs
        st.success("Notes directories updated!")
        # Force reinitialization of the index
        st.cache_resource.clear()
        st.session_state.index = initialize_index()

# Main query interface
st.header("Chat with Your Documents")

# Display chat history
for i, (role, message) in enumerate(st.session_state.chat_history):
    with st.chat_message(role):
        st.write(message)

# Query input and model selection
col1, col2 = st.columns([3, 1])
with col1:
    query = st.chat_input("Enter your query:")
with col2:
    model_name = st.selectbox(
        "Model:",
        ["gemini-1.5-flash", "gpt-4o", "gpt-4o-mini"],
        index=0,
        label_visibility="collapsed"
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

# Display model selection information
st.caption(f"Current model: {model_name}")
