from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}'),
	MessagesPlaceholder(variable_name='userpref')
])

userpref = "I prefer to receive updates via email."
# chat_history = []
chat_history = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='hi'),
    AIMessage(content='hello, how can I help you'),
    HumanMessage(content='I have a question about my order'),
    AIMessage(content='sure, what is your order number?'),
    HumanMessage(content='12345'),
    AIMessage(content='thank you, I see that your order has been shipped and should arrive tomorrow.')
]
# load chat history
# with open('chat_history.txt') as f:
#     chat_history.extend(f.readlines())

# print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my order?', 'userpref':userpref})

print(prompt)

# now u have to pass the prompt which u have generated above to the model and get the response.