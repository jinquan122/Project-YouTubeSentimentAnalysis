import streamlit as st
from app.task.taskd_indepth_discussion import InDepthDiscussion

def main():
    st.title("CrewAI - InDepth Discussion")
    st.caption("This app allows you to ask complex questions regarding the researched product. The agent will answer your query using youtube sentiment and data from external website.")
    input = st.text_input("Enter a query...")
    if input:
        with st.spinner("Running analysis..."):
            indepth_discussion = InDepthDiscussion(input, st.session_state['subject'])
            analysis_result = indepth_discussion.run()
            st.write(analysis_result)

if __name__ == "__main__":
    main()