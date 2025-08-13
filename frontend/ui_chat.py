import streamlit as st

def render_chat_settings():
    """Renders the settings for the chat, like dataset and strategy selection."""
    if "rag_strategies" not in st.session_state:
        try:
            st.session_state.rag_strategies = st.session_state.api_client.get_rag_strategies()
        except Exception as e:
            st.error(f"Failed to load RAG strategies: {e}")
            st.session_state.rag_strategies = []

    # Prepare dataset options for multiselect
    datasets = st.session_state.get("datasets", [])
    dataset_options = {d["name"]: d["id"] for d in datasets}

    with st.expander("Chat Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset_names = st.multiselect(
                "Select Datasets to Query",
                options=dataset_options.keys(),
                default=st.session_state.get("selected_dataset_names", [])
            )
            st.session_state.selected_dataset_names = selected_dataset_names
            st.session_state.selected_dataset_ids = [dataset_options[name] for name in selected_dataset_names]
        with col2:
            selected_strategy = st.selectbox(
                "Select RAG Strategy",
                options=st.session_state.rag_strategies,
                index=st.session_state.rag_strategies.index(st.session_state.get("selected_strategy", "basic"))
                if st.session_state.get("selected_strategy") in st.session_state.rag_strategies else 0
            )
            st.session_state.selected_strategy = selected_strategy

def render_chat():
    st.header("Chat with your documents")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.info(f"**Source:** `{source.get('metadata', {}).get('original_filename', 'N/A')}`\n\n---\n{source.get('page_content', '')}")

    # Render chat settings before the input
    render_chat_settings()

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    dataset_ids = st.session_state.get("selected_dataset_ids")
                    strategy = st.session_state.get("selected_strategy")

                    if not dataset_ids:
                        st.warning("Please select at least one dataset in 'Chat Settings'.")
                        st.session_state.messages.pop() # Remove user prompt
                        return

                    response = st.session_state.api_client.query_rag(
                        query=prompt,
                        dataset_ids=dataset_ids,
                        strategy=strategy
                    )
                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    # Use st.experimental_rerun() for older Streamlit, st.rerun() is fine for newer versions
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.pop() # Remove user prompt on error
