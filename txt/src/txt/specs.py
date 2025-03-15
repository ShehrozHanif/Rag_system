from crewai import Agent, Task, Crew,LLM, Process
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

llm1 = LLM(model="gemini/gemini-2.0-flash")


# Create agent-specific knowledge about a product
product_specs = StringKnowledgeSource(
    content="""The XPS 13 laptop features:
    - 13.4-inch 4K display
    - Intel Core i7 processor
    - 16GB RAM
    - 512GB SSD storage
    - 12-hour battery life""",
    metadata={"category": "product_specs"}
)

google_embedder = {
    "provider": "google",
    "config": {
         "model": "models/text-embedding-004",
         "api_key":"AIzaSyASKilFqlcEti-u2SapRbKF04N7xBvVX6k",
         }
}


# Create a support agent with product knowledge
support_agent = Agent(
    role="Technical Support Specialist",
    goal="Provide accurate product information and support.",
    backstory="You are an expert on our laptop products and specifications.",
    verbose=True,
    allow_delegation=False,
    llm=llm1,
    # knowledge_sources=[product_specs]  # Agent-specific knowledge
)

# Create a task that requires product knowledge
support_task = Task(
    description="Answer this customer question: {question}",
    agent=support_agent,
    expected_output="A clear and concise answer about the XPS 13 laptop's specifications."
)



crew = Crew(
    memory=True,
    agents=[support_agent],
    tasks=[support_task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[product_specs], # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
    embedder=google_embedder

)



# Get answer about the laptop's specifications

def main():
    result = crew.kickoff(
        inputs={"question": "What is the storage capacity of the XPS 13?"}
    )
    print(result)