import streamlit as st 
from helper import chatwithsqldatabase

st.title("Chat with SQL Database")
with st.form("db_form"):
    db_user = st.text_input("Database User", placeholder="Enter your database username")
    db_password = st.text_input("Database Password", type="password", placeholder="Enter your database password")
    db_host = st.text_input("Database Host", placeholder="Enter your database host")
    db_name = st.text_input("Database Name", placeholder="Enter your database name")
    
    # Submit button to connect to the database
    submitted = st.form_submit_button("Connect")
    
if submitted and 'chat_agent' not in st.session_state:
    try:
        st.session_state.chat_agent = chatwithsqldatabase(db_user=db_user, db_password=db_password, db_host=db_host, db_name=db_name)
        st.success(f"Connected to the database `{db_name}` on host `{db_host}`!")
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
    

if "chat_agent" in st.session_state:
    st.write("Now you can query the database!")

    query = st.text_area("Enter your question related to data")

    if st.button("Run Query"):
        try:
            question, new_query,sql_answer, ai_answer= st.session_state.chat_agent.message_to_sql_helper(query)
            st.write("SQL Results:", sql_answer)
            st.write("Query Result:", ai_answer)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
