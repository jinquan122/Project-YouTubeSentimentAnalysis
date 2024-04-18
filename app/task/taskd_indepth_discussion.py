from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import Tool
from crewai import Agent, Task, Crew
from app.genai.llm import get_langchain_llm
from langchain.pydantic_v1 import BaseModel, Field
from app.task.taskc_searching_function import search
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional, Type

class LanceDBsearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    
class LanceDBsearchTool(BaseTool):
    name:str = "LanceDB search"
    description:str = "Useful when search for reviews in LanceDB."
    args_schema:Type[BaseModel] = LanceDBsearchInput

    def _run(self, query:str, run_manager = None) -> str:
        nodes_text, nodes_sentiment, nodes_score = search(query)
        print("Lancedb serach:", query)
        return str(nodes_text)

class InDepthDiscussion():
    def __init__(self, query:str, product:str):
        self.llm = get_langchain_llm()
        self.query = query
        self.product = product

    def _search_tool(self):
        '''
        Define a tool for searching.

        Returns:
            Tool: The tool for searching.
        '''
        search = DuckDuckGoSearchResults()
        search_tool = Tool(
        name="DuckDuckGo_search_tool",
        func=search.run,
        description="Useful for performing search to get extra information on subject when the subject information is not enough.",
        )
        return search_tool
    
    def _sentiment_search_tool(self):
        sentiment_search_tool = LanceDBsearchTool()
        return sentiment_search_tool

    def _define_agent(self):
        '''
        Define the agents for the task.

        Returns:
            Agent: The agents for the task: researcher and senior researcher.
        '''
        researcher = Agent(
            role='Researcher',
            goal=f'Find the reason or terminology behind of the query.',
            backstory='An expert researcher with a keen eye for market trends.',
            llm=self.llm,
            verbose=True
        )

        senior_researcher= Agent(
            role='Senior Researcher',
            goal=f'Further improve findings from researcher.',
            backstory='An experienced researcher with a deep understanding of market research and data analyst skill.',
            llm=self.llm,
            verbose=True
        )
        return researcher, senior_researcher

    def _define_tasks(self, researcher, senior_researcher, search_tool, sentiment_search_tool):
        '''
        Define the tasks for the task.

        Returns:
            Task: The tasks for the task: query research and enhanced research.
        '''
        query_research = Task(
            description=f'Perform a research on the given query to find reason or terminology behind based on reviews in LanceDB and extra information. Target subject is {self.product}. Query: {self.query}',
            expected_output=f'A comprehensive analysis.',
            agent=researcher,
            tools=[sentiment_search_tool, search_tool]
        )

        enhanced_research = Task(
            description=f'Perform sencond level of research on the analysis done by researcher based on given query. Target subject is {self.product}. Query: {self.query}',
            expected_output=f'A comprehensive enhanced analysis.',
            agent=senior_researcher,
            tool=[search_tool]
        )
        return query_research, enhanced_research
    
    def _define_crew(self, query_research, enhanced_research, researcher, senior_researcher):
        '''
        Define the crew for the task.

        Returns:
            Crew: The crew which have their own tasks and agents.
        '''
        crew = Crew(
                    agents=[researcher, senior_researcher],
                    tasks=[query_research, enhanced_research],
                    verbose=2
                )
        return crew
    
    def run(self):
        '''
        Run the In depth discussion.

        Returns:
            str: The analysis of the discussion.
        '''
        search_tool = self._search_tool()
        sentiment_search_tool = self._sentiment_search_tool()
        researcher, senior_researcher = self._define_agent()
        query_research, enhanced_research = self._define_tasks(researcher, senior_researcher, search_tool, sentiment_search_tool)
        crew = self._define_crew(query_research, enhanced_research, researcher, senior_researcher)
        analysis = crew.kickoff()
        return analysis