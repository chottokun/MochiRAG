import streamlit as st

def render_chat():
    st.header("Chat with your documents")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.info(f"**Source:** `{source['metadata']['original_filename']}`\n\n---\n{source['page_content']}")

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    dataset_id = st.session_state.get("selected_dataset_id")
                    if not dataset_id:
                        st.warning("Please select a dataset from the sidebar first.")
                        return

                    response = st.session_state.api_client.query_rag(
                        query=prompt,
                        dataset_ids=[dataset_id]
                    )
                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {e}")
