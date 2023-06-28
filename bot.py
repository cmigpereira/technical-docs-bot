import os
import pandas as pd
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

os.environ['OPENAI_API_KEY'] = 'XXXXXXXXXXXXXXXXXX'

st.set_page_config(
    page_title="BOT - Demo",
    page_icon="🤖"
)

st.sidebar.title('BOT 🤖 - A Bot for Optimizely\'s Technical documentation')
st.sidebar.write("""
         ###### A Q&A chatbot for Optimizely technical documentation using GPT-3 developed by Carlos Pereira.
         """)

def get_answer(question, search_index, chain):
    '''
    Get answer from GPT
    '''
    answer = chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    return answer

def app():
    '''
    The main app
    '''

    user_input = st.text_input('Do you have a question for me?')
    search_index = FAISS.load_local("faiss_search_index", OpenAIEmbeddings())
    chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if user_input:
        output = get_answer(user_input, search_index, chain)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], is_user=False, avatar_style="bottts", key=str(i))
            message(st.session_state['past'][i], is_user=True, avatar_style="adventurer", key=str(i) + '_user')

if __name__ == '__main__':
    app()
