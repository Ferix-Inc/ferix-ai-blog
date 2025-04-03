import os
import boto3
import copy
from pathlib import Path
from typing_extensions import TypedDict, Annotated
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_aws import ChatBedrockConverse
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode
from utils import generate_quotation, text_to_embedding


PROFILE = os.getenv("AWS_PROFILE", "default")
MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
REGION = "ap-northeast-1"

FILE_FORMATS = [".pdf", ".xlsx"]
PAST_QUOTATIONS_DIR = Path("past_quotations")

ENABLE_ACTIONS = (
    "   - Use the internet to improve the quotation\n"
    "   - Compare with past quotations to improve the quotation\n"
)

session = boto3.Session(profile_name=PROFILE)
client = session.client("bedrock-runtime", region_name=REGION)
llm = ChatBedrockConverse(client=client, model=MODEL)
embedding = BedrockEmbeddings(client=client)
vector_store = InMemoryVectorStore(embedding)
memory = MemorySaver()


class State(TypedDict):
    query: str
    project_files: list[str]
    reference_files: list[str]
    past_quotations: list[Document]
    goal: str
    quotation: str
    messages: Annotated[list, add_messages]


def setup(state: State) -> State:
    """
    The first node in the workflow. It performs the following tasks:
      1. Generates an initial quotation from project files. If reference files exist,
         it first creates a quotation from them and then uses it to guide a new quotation.
      2. Generates a 'goal' statement using an LLM based on the user's query.
      3. Loads past quotation documents from the specified directory.

    Args:
        state: The initial workflow state.

    Returns:
        A dictionary that updates 'quotation', 'goal', 'past_quotations', and 'messages'.
    """
    quotation = _initial_quotation(state)
    goal = _goal(state)
    past_quotations = _past_quotations(state)
    message = HumanMessage(
        f"You are a kindful assistant. Please help me.\ngoal: {goal}"
    )
    vector_store.add_documents(past_quotations)

    return {
        "quotation": quotation,
        "goal": goal,
        "past_quotations": past_quotations,
        "messages": [message],
    }


def _initial_quotation(state: State) -> str:
    """
    Generates an initial quotation based on the provided project files.
    If reference files are present, it first creates a quotation from them and then uses
    that result to guide a new quotation from the project files.

    Args:
        state: The current workflow state.

    Returns:
        A JSON string representing the initial quotation adhering to the Project schema.
    """
    if state.get("reference_files"):
        reference_quotation = generate_quotation(
            client,
            prompt="Please create a quotation based on the provided documents.",
            files=state["reference_files"],
        )
        quotation = generate_quotation(
            client,
            prompt=(
                "Please refer to the following quotation and create a new one from the project files.\n\n"
                f"Past quotation:\n{reference_quotation}\n"
            ),
            files=state["project_files"],
        )
    else:
        quotation = generate_quotation(
            client,
            prompt="Please create a quotation based on the provided documents.",
            files=state["project_files"],
        )
    return quotation


def _goal(state: State) -> str:
    """
    Reads the user's query and reformulates it into a concise, actionable goal
    using the LLM. Only the actions listed in ENABLE_ACTIONS may be considered.

    Args:
        state: The current workflow state.

    Returns:
        A string that describes a clear, feasible goal derived from the user's query.
    """
    prompt = (
        "You will receive a user request to create or improve a quotation. "
        "Please produce a clear and actionable goal.\n"
        "Requirements:\n"
        "1. The goal must be concrete, specific, and feasible.\n"
        f"2. Your available actions are strictly limited to:\n{ENABLE_ACTIONS}\n\n"
        f"User input: {state['query']}"
    )
    response = llm.invoke(prompt)
    return response.content


def _past_quotations(state: State) -> list[Document]:
    """
    Scans the PAST_QUOTATIONS_DIR for subdirectories, each presumably representing a past project.
    For each directory containing recognized file formats, it creates a quotation and stores it as a Document.

    Args:
        state: The current workflow state.

    Returns:
        A list of Document objects, each containing a past quotation in JSON format.
    """
    past_quotations = []
    if not PAST_QUOTATIONS_DIR.exists():
        return []
    for project_dir in PAST_QUOTATIONS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        files = [f for f in project_dir.iterdir() if f.suffix in FILE_FORMATS]
        if files:
            q = generate_quotation(
                client,
                prompt="Please create a quotation based on the provided documents.",
                files=files,
            )
            document = Document(
                page_content=q,
                metadata={"project_name": project_dir.name, "files": files},
            )
            past_quotations.append(document)
    return past_quotations


@tool
def improve_quotation_with_internet(state: State) -> State:
    """
    Improves the current quotation by conducting a simple internet search
    using TavilySearchResults and applying insights from the search to refine the quotation.

    Args:
        state: The current workflow state.

    Returns:
        A dictionary with an updated 'quotation' and a new message describing the changes.
    """
    plan = state["messages"][-1].content
    search_query_prompt = (
        "Based on the plan below, please come up with a short query to search on the internet:\n"
        f"quotation: {state['quotation']}\n"
        f"plan: {plan}"
    )
    search_query = llm.invoke(search_query_prompt).content

    tool = TavilySearchResults(
        max_results=1,
        include_answer=False,
        include_raw_content=True,
    )
    response = tool.invoke({"query": search_query})
    search_results = "\n".join(c["content"] for c in response)

    # Improve the quotation based on the search results
    prompt = (
        "Please improve the following current quotation based on the search results:\n\n"
        f"Current quotation:\n{state['quotation']}\n\n"
        f"Plan : {plan}\n\n"
        f"Search query: {search_query}\n\n"
        f"Search resluts:\n{search_results}"
    )
    quotation = generate_quotation(client, prompt=prompt)
    return {
        "quotation": quotation,
        "messages": [
            AIMessage("Improved the quotation using internet search results.")
        ],
    }


@tool
def improve_quotation_with_past_comparisons(state: State) -> State:
    """
    Compares the current quotation to similar past quotations and refines it based on best practices
    identified in those past quotations.

    Args:
        state: The current workflow state.

    Returns:
        A dictionary with an updated 'quotation' and a new message describing the improvements.
    """
    current_quotation_embedding = text_to_embedding(client, state["quotation"])
    docs = vector_store.similarity_search(current_quotation_embedding, k=3)
    similar_quotations = "\n\n".join(
        [
            f"project name: {doc.metadata['project_name']}\ncontent:\n{doc.page_content}"
            for doc in docs
        ]
    )

    # Improve the quotation based on past examples
    prompt = (
        "Please improve the following current quotation based on similar past project quotations:\n\n"
        f"Current quotation:\n{state['quotation']}\n\n"
        f"Plan : {state['messages'][-1].content}\n\n"
        f"Similar past quotations:\n{similar_quotations}\n\n"
    )
    quotation = generate_quotation(client, prompt=prompt)

    return {
        "quotation": quotation,
        "messages": [
            AIMessage("Improved the quotation using similar past quotations.")
        ],
    }


def agent(state: State) -> State:
    """
    - Generate a plan with LLM based on the user's query.
    - Hand off the decision of which tool to call or to end to route_tools().
    """

    messages = copy.deepcopy(state["messages"])
    # 1. generate plan
    prompt = (
        "Please plan the single next action necessary for improvement.\n"
        f"current quotation: {state['quotation']}\n\n"
        f"Allowed actions:\n{ENABLE_ACTIONS}"
    )
    messages += [HumanMessage(prompt)]
    plan = llm.invoke(messages).content
    messages += [AIMessage(plan)]

    # 2. user reflection
    print(f"next plan:\n{plan}")
    feed_back = interrupt("Please provide feedback on the plan.")

    # 3. regenerate plan
    prompt = (
        "Please plan the single next action necessary for improvement.\n"
        f"current quotation: {state['quotation']}\n\n"
        f"Allowed actions:\n{ENABLE_ACTIONS}\n\n"
        f"plan: {plan}\n"
        f"feedback: {feed_back}"
    )
    messages += [HumanMessage(prompt)]
    response = llm.bind_tools(
        [improve_quotation_with_past_comparisons, improve_quotation_with_internet]
    ).invoke(messages)

    return {"messages": [HumanMessage(prompt), response]}


def route_tools(state: State) -> str:
    """
    Determine whether to:
      - use tools to improve the quotation
      - or end the process
    """

    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END


def build_quotation_workflow():
    graph_builder = StateGraph(State)
    tool_node = ToolNode(
        tools=[improve_quotation_with_internet, improve_quotation_with_past_comparisons]
    )
    graph_builder.add_node("setup", setup)
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tool", tool_node)
    graph_builder.add_edge(START, "setup")
    graph_builder.add_conditional_edges(
        "agent", route_tools, {"tools": "tools", END: END}
    )
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def main():
    graph = build_quotation_workflow()
    query = "見積りを改善してください。"
    project_files = [
        "past_quotations/project_1/quotation.xlsx",
    ]

    input = {
        "query": query,
        "project_files": project_files,
    }
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream(input, thread, stream_mode="updates"):
        print(event)

    for event in graph.stream(
        Command(resume="looks good."), thread, stream_mode="updates"
    ):
        print(event)


if __name__ == "__main__":
    main()
