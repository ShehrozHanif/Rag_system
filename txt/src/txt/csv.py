from crewai import Agent, Task, Crew, Process, LLM



llm1 = LLM(model="gemini/gemini-2.0-flash")

# Create a knowledge source
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource



# Create a PDF knowledge source
pdf_source = CSVKnowledgeSource(
    file_paths=["document-csv.csv"],
 
)


google_embedder = {
    "provider": "google",
    "config": {
         "model": "models/text-embedding-004",
         "api_key":"AIzaSyBS_AzNTJ7hicY3FG_HWFefZaaljnak_DA",
         }
}

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=llm1,
)
task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    memory=True,
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[pdf_source], # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
    embedder=google_embedder

)
def main():
    result = crew.kickoff(inputs={"question": "Which Job Shehroz Hanif Can Do?"})
    with open("csv_output.csv", "w") as f:
        f.write(str(result))
    print("Done! Check the csv_output.csv file for the result.")