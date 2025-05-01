import streamlit as st
from helper import stream_data
from helper import chatwithsqldatabase

st.title("Chat with SQL Database")
with st.sidebar.form("db_form"):
    db_user = st.text_input(
        "Database User", placeholder="Enter your database username")
    db_password = st.text_input(
        "Database Password", type="password", placeholder="Enter your database password")
    db_host = st.text_input(
        "Database Host", placeholder="Enter your database host")
    db_name = st.text_input(
        "Database Name", placeholder="Enter your database name")

    submitted = st.form_submit_button("Connect")

if submitted and 'chat_agent' not in st.session_state:
    try:
        st.session_state.chat_agent = chatwithsqldatabase(
            db_user=db_user, db_password=db_password, db_host=db_host, db_name=db_name)
        st.success(
            f"Connected to the database `{db_name}` on host `{db_host}`!")
        
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")


if "chat_agent" in st.session_state:
    st.success("✅ Now you can query the database!")
else:
    st.warning("⚠️ Please connect to the database first.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["generated_sql_query"])
            st.markdown(message["generated_sql_answer"])
            st.markdown(message["content"])
            
        

if query := st.chat_input("Enter your SQL question"):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        with st.spinner("Generating SQL query..."):
            question, new_query, sql_answer, prompt_2 = st.session_state.chat_agent.message_to_sql_helper(query)
            new_query = f"**Generated SQL Query:**\n```sql\n{new_query}\n```"
            sql_answer = f"**SQL Results:**\n{sql_answer}"
            if "Error:" in str(sql_answer):
                st.warning(new_query)
                st.warning(sql_answer)
                st.warning("⚠️ SQL error encountered. Sending back to AI for correction...")
                correction_prompt = f"""
                The following SQL query was generated from the question: "{query}".

                Generated SQL:
                {new_query}

                This error occurred during execution:
                {sql_answer}

                Please fix the SQL query based on this error and provide a corrected query that will work properly.
            """

                corrected_result = st.session_state.chat_agent.correct_sql_with_error(correction_prompt)

                question, new_query, sql_answer, prompt_2 = corrected_result
                new_query = f"**Generated SQL Query:**\n```sql\n{new_query}\n```"
                sql_answer = f"**SQL Results:**\n{sql_answer}"

            
            with st.chat_message("assistant"):
                st.markdown(new_query)
                st.markdown(sql_answer)

        with st.spinner("Generating AI answer..."):
            ai_answers = st.session_state.chat_agent.generate_final_tokens(prompt_2)
            st.write_stream(stream_data(ai_answers['response']))
            st.session_state.messages.append({
                "role": "assistant",
                "generated_sql_query": new_query,
                "generated_sql_answer": sql_answer,
                "content": ai_answers['response']
            })

    except AttributeError:
        st.error("❌ Database connection not found. Please connect to the database using the sidebar first.")
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
