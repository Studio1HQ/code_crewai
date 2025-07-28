import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileWriterTool

# === Tools ===
file_writer_tool = FileWriterTool()

# === LLMs for each agent ===
architect_llm = LLM(
    model="novita/moonshotai/kimi-k2-instruct",
    temperature=0.5,
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ['NOVITA_API_KEY']
)

coder_llm = LLM(
    model="novita/qwen/qwen3-coder-480b-a35b-instruct",
    temperature=0.4,
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ['NOVITA_API_KEY']
)

reviewer_llm = LLM(
    model="novita/moonshotai/kimi-k2-instruct",
    temperature=0.5,
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ['NOVITA_API_KEY']
)

# === Agent Definitions ===

architect = Agent(
    role="Software Architect for MVP Projects",
    goal="Define a basic system structure and simple feature set to guide MVP development",
    backstory="""You specialize in quickly outlining software architectures and project scopes for 
minimum viable products. Your plans help guide coders with enough structure to get started while staying lean.""",
    llm=architect_llm,
    verbose=True
)

coder = Agent(
    role="Developer for MVP Projects",
    goal="Implement the MVP using simple, clear code, and write all necessary code files using the FileWriter tool",
    backstory="""You're a practical developer focused on speed. You prioritize working code over polish. 
Simulate any runtime behavior if needed, and make sure to keep code clear and modular.""",
    llm=coder_llm,
    verbose=True,
    tools=[file_writer_tool]
)

reviewer = Agent(
    role="Code Reviewer for MVP Projects",
    goal="Read the code files, provide helpful feedback, and write improvements as diffs into a markdown file",
    backstory="""You're a fast but thoughtful reviewer. You check code for clarity, obvious bugs, and 
provide improvements as diffs. Instead of modifying code directly, you save your suggestions in a markdown file 
for easy inspection.""",
    llm=reviewer_llm,
    verbose=True,
    tools=[file_writer_tool]
)

# === Tasks ===

architect_task = Task(
    description="""
Create a basic plan for building a simple MVP version of a {project}.
Focus on the core features, basic file structure, and tech stack.
Make the structure minimal but enough to get a working demo.
""",
    expected_output="""
A simple architectural overview with:
- Project goals
- Key components or files
- Basic data flow or structure
""",
    agent=architect,
    output_file="architecture.md"
)

coder_task = Task(
    description="""
Based on the architect's plan, implement the core parts of the {project} MVP.
Keep it lean and functional. Use the FileWriter tool to save all your code files.
If you need to simulate behavior (e.g. without a real interpreter), do so with clear comments or mocked logic.
""",
    expected_output="""
Working code files saved using the FileWriter tool that implement the key features defined in the plan.
Code should be readable and logically structured.
""",
    agent=coder,
    context=[architect_task]
)

review_task = Task(
    description="""
Review the code files created for the {project} MVP. Read each file using the FileRead tool.
Suggest improvements using diffs. Do not modify the original code files directly.
Instead, save all your suggested diffs into a markdown file using the FileWriter tool.
""",
    expected_output="""
A markdown file containing diff-style suggestions for each file reviewed.
""",
    agent=reviewer,
    context=[coder_task],
    output_file="code_review_diffs.md"
)

# === Crew Definition ===

code_crew = Crew(
    agents=[architect, coder, reviewer],
    tasks=[architect_task, coder_task, review_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    code_crew.kickoff(inputs={"project": "Todo App"})
