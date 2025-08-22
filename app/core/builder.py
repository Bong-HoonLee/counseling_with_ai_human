def build_graph():
    from langgraph.graph import START, END, StateGraph
    from node import GraphState, Chatbot_node, is_retrieve
    
    workflow = StateGraph(GraphState)
    chatbot_node = Chatbot_node("openai_gpt")

    # add node
    workflow.add_node("retrieve", chatbot_node.retrieve)
    workflow.add_node("rag_response", chatbot_node.rag_response)
    workflow.add_node("gpt_relevance_check", chatbot_node.relevance_check)
    workflow.add_node("conversation_response", chatbot_node.conversation_response)
    
    # add edge
    workflow.add_edge(START, "retrieve")
    # workflow.add_edge("rag_response", "gpt_relevance_check")
    # workflow.add_edge("gpt_relevance_check", END)
    workflow.add_conditional_edges(
        source="retrieve",
        path=is_retrieve,
        path_map={"rag": "rag_response", 'conversation': 'conversation_response'},
    )
    
    workflow.add_edge("rag_response", END)
    workflow.add_edge("conversation_response", END)
    
    return workflow