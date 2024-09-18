import os
import gradio as gr
from dotenv import load_dotenv
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.Conversation import Conversation
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete import SystemMessage

# Add API key to environment variable.
load_dotenv()

# Get the API_KEY
API_KEY = os.getenv("GROQ_API_KEY")

# Create an instance of GroqModule
llm = GroqModel(api_key=API_KEY)

# Get available model from the instance created above
allowed_models = llm.allowed_models

# Create a SimpleConversationAgent with the GroqModel
agent = SimpleConversationAgent(llm=llm, conversation=Conversation())

#  Initialize a MaxSystemContexConversation instance
conversation = MaxSystemContextConversation()


# Define a function to dynamically change the model based on dropdown input
def load_model(selected_model):
    return GroqModel(api_key=API_KEY, name=selected_model)


# Define the function to be executed for the gradio interface

def chat_agent(input_text, history, system_context, model_name):
    #   initialize the model dynamically based on user selection
    llm = load_model(model_name)

    # Initializing agent with the new model
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)

    agent.conversation.system_context = SystemMessage(content=system_context)

    # Get input and convert to string
    input_text = str(input_text)

    # Execute the input command with the agent
    result = agent.exec(input_text)

    # Return the result as a string
    return str(result)


# Creating a user interface
demo_chat_agent = gr.ChatInterface(fn=chat_agent, additional_inputs=[
    gr.Textbox(label="System Context"),
    gr.Dropdown(label="Model Name", choices=allowed_models, value=allowed_models[0])
],
                                   title="Chat Agent",
                                   description="Select a model to begin chatting."
                                   )
# Run the chat application
if __name__ == "__main__":
    demo_chat_agent.launch()
