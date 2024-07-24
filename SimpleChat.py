import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Parameters
temperature = 1
max_length = 400
llm_model = 'tiiuae/falcon-7b-instruct'
token = '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'

# Load LLM
llm = HuggingFaceHub(
    repo_id=llm_model,
    model_kwargs={'temperature': temperature, 'max_length': max_length},
    huggingfacehub_api_token=token
)

# Initialize chat message history
history = StreamlitChatMessageHistory(key="chat_messages")

# Initialize memory
memory = ConversationBufferMemory(chat_memory=history)

# Define the prompt template
template = """You are one of the smartest AI chatbots who answer all questions of humans and gives response.
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["human_input"], template=template)

# Initialize the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit app
st.title('Welcome to the ChatBot')

# Display previous messages from history
for msg in history.messages:
    if msg.type == "human":
        st.chat_message("human").write(msg.content)
    elif msg.type == "ai":
        # Process the stored response to ensure it only displays the AI's answer
        response_lines = msg.content.split('\n')
        ai_response = None
        for line in response_lines:
            if line.startswith('AI:'):
                ai_response = line[len('AI: '):].strip('\"')
                break
            
        if not ai_response:
            ai_response = "Sorry, no idea."
            
        st.chat_message("ai").write(ai_response)

# Handle new input
if user_input := st.chat_input():
    # Display user message
    st.chat_message("human").write(user_input)

    # Generate AI response
    response = llm_chain.run(user_input)
    # Split the response by lines
    lines = response.split('\n')
    ai_response = None
    # Iterate through the lines to find the line starting with 'AI:'
    for line in lines:
        if line.startswith('AI:'):
            # Extract the response
            ai_response = line[len('AI: '):].strip('\"')
            break
    
    # Display AI response if not null
    if response.strip():
        if not ai_response:
            ai_response = "Sorry, no idea."
            
        st.chat_message("ai").write(ai_response)