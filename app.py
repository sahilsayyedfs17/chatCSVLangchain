import streamlit as st
import pandas as pd
import openai
import os
import re
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from apikey import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY),
        file_path, 
        verbose=True,
        allow_dangerous_code=True,  
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        response = agent.run(user_message)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Display content to Streamlit based on the structure of the provided JSON.
    """
    
    # Check if the response has plain text.
    if "answer" in json_response:
        st.write(json_response["answer"])

    # Check if the response has a bar chart.
    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response has a table.
    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""

    st.title('CSV Assistant')
    st.write('Please upload your CSV file and enter your query below:')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Save the uploaded file to disk
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)
        
        user_input = st.text_input("Your query")
        if st.button('Run'):
            response = csv_agent_func(file_path, user_input)
            
            if response:
                st.write("Agent response:")
                st.write(response)
                
                # Extracting code from the response
                code_to_execute = extract_code_from_response(response)
                
                if code_to_execute:
                    st.write("Extracted code to execute:")
                    st.code(code_to_execute)
                    try:
                        # Making df available for execution in the context
                        exec(code_to_execute, globals(), {"df": df, "plt": plt, "st": st})
                        fig = plt.gcf()  # Get current figure
                        st.pyplot(fig)  # Display using Streamlit
                    except Exception as e:
                        st.write(f"Error executing code: {e}")
                else:
                    st.write("No executable code found in the response.")
            else:
                st.write("No response from agent.")

    st.divider()
            
if __name__ == "__main__":
    csv_analyzer_app()
