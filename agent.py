import warnings
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Ignorer l'avertissement de dépréciation
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Définir l'outil
@tool
def multiplier(a: int, b: int) -> int:
    """Multiplie deux nombres entiers."""
    return a * b

# 2. Initialiser le modèle
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

# 3. Créer l'agent (la méthode qui fonctionne sur votre machine)
tools = [multiplier]
agent_executor = create_react_agent(llm, tools)

# 4. Tester
if __name__ == "__main__":
    events = agent_executor.stream(
        {"messages": [("human", "Combien font 123 multiplié par 456 ?")]},
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            last_message = event["messages"][-1]
            if last_message.type == "ai" and last_message.content:
                print("\nRéponse :", last_message.content
