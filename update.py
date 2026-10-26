import streamlit as st
st.write(st.__version__)

if hasattr(st, "autorefresh"):
    st.write("✅ autorefresh is available!")
else:
    st.write("❌ autorefresh not found. Update Streamlit.")
