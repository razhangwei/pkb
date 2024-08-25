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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Index"):
            main.update_index(st.session_state.index, st.session_state.notes_directories)
            st.success("Index updated successfully!")
            # Force reinitialization of the index
            st.cache_resource.clear()
            st.session_state.index = initialize_index()
    with col2:
        if st.button("Rebuild Index"):
            # Clear the existing index
            st.cache_resource.clear()
            # Rebuild the index from scratch
            documents = []
            for directory in st.session_state.notes_directories:
                documents.extend(main.load_documents(directory))
            st.session_state.index = main.create_index(documents, verbose=True)
            main.save_index(st.session_state.index)
            st.success("Index rebuilt successfully!")
    
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
        
        # Update NOTES_DIRECTORIES environment variable in .env file
        with open('.env', 'r') as f:
            lines = f.readlines()
        with open('.env', 'w') as f:
            for line in lines:
                if line.startswith('NOTES_DIRECTORIES='):
                    f.write(f'NOTES_DIRECTORIES={",".join(new_dirs)}\n')
                else:
                    f.write(line)
        
        # Rebuild the index from scratch
        st.cache_resource.clear()
        documents = []
        for directory in st.session_state.notes_directories:
            documents.extend(main.load_documents(directory))
        st.session_state.index = main.create_index(documents, verbose=True)
        main.save_index(st.session_state.index)
        st.success("Index rebuilt with updated directories!")

# Main query interface
st.header("Chat with Your Documents")

# Create a container for the chat history
chat_container = st.container()

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

# Display chat history in the container
with chat_container:
    for i, (role, message) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.write(message)

if query:
    # Add user message to chat history
    st.session_state.chat_history.append(("user", query))
    with chat_container:
        with st.chat_message("user"):
            st.write(query)

    # Prepare context from chat history
    context = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history[-5:]])

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                response = main.query_index(st.session_state.index, query, model_name, context)
            st.write(response)

    # Add assistant response to chat history
    st.session_state.chat_history.append(("assistant", response))

# Display model selection information
st.caption(f"Current model: {model_name}")
