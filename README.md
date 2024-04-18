# YouTube Advanced Sentiment Analysis

## Inspiration
Sentiment analysis is not only being performed in articles or review sections. It can be in video format like YouTube video. Short video and video are the trends or information source for our generation instead of reading articles or newspaper. Hence, we cannot deny the information conveyed in videos. 

## What it does
The apps perform sentiment analysis from YouTube video content. There will be three different parts for the apps:
1. Perform sentiment analysis on YouTube video content and display the summary and result on a dashboard. (Provide YouTube link for the video source)
2. Provide sentiment sentences search function from our temporary database about the sentiment sentences from YouTube video content.
3. Indepth discussion function for asking complex questions.

## How we built it
Apps Tech Stack:
1. Gemini pro API
2. Gemini embedding-001
3. LanceDB (vector store)
4. LangChain and LlamaIndex (RAG framework)
5. CrewAI
6. Streamlit
7. Main Python libraries: Scikit-learn, Scipy, YouTubeSearchTool, YouTubeTranscriptApi, Wikipedia

Part 1:
1. Extract YouTube video and extract the transcript content using the YouTube relevant Python libraries.
2. Extract structured positive and negative reviews using LangChain and Gemini Pro.
3. Topic Modeling - Perform sentiment clustering using hierarchical clustering and generate topic using Gemini Pro.
4. The data are displayed into the dashboard.

Part 2:
1. All the extracted sentiments from Part 1 will be stored into LanceDB (vector database) by using Gemini Embedding model.
2. The positive and negative sentiment are stored into different tables in LanceDB.
3. Define retriever strategy using LlamaIndex.
4. The result will be filtered by a similarity threshold of 0.5 (or set by user).

Part 3:
1. Define two AI workforces (Gemini Pro) using CrewAI: Researcher and Senior Researcher.
2. Both of them are assigned with the tools to search using DuckDuckGo searching tool and LanceDB sentiment searching tool.
3. Define the tasks they need to carry out during the AI workforce discussion session.
4. Generate discussion results based on the complex question asked by user.

## Challenges we ran into
The consistency of YouTube video as the knowledge source are not good. Hence, the sentiment result might vary due to different of videos as the knowledge source.

## Accomplishments that we're proud of
We successfully summarize the YouTube video content and provide a tidy and clean interface for user to dive deeper without many efforts.

## What we learned
We learned prompt engineering may not be a perfect way to make sure the LLM output consistency. To cope with that problem, we decided to use Gemini as a micro service for each separate function. Gemini tends to have a better output if the task is parsed into simpler.

## What's next for YouTube Advanced Sentiment Analysis
We might look into Use of multi modal LLM to understand audio or video file instead of transcript file. (Including short video which is a trend now)
