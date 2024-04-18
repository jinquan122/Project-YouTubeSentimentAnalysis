import ast
from langchain_community.tools import YouTubeSearchTool
import re
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from app.genai.llm import get_langchain_llm
from app.genai.prompt import structured_sentiment_prompt, get_extract_sentiment_prompt
from tqdm import tqdm


class YouTubeSentimentExtractor():
    def __init__(self, product:str):
        self.product = product
        self.llm = get_langchain_llm()
        
    def _extract_videoID(self, youtube_link:str) -> str:
        '''
        Extracts the video ID from the YouTube link.

        Args:
            youtube_link (str): The YouTube link.

        Returns:
            str: The video ID.
        '''
        pattern = r'v=([^&]+)&pp'
        match = re.search(pattern, youtube_link)
        if match:
            return match.group(1)
        else:
            return None
    
    def _youtube_search_tool(self):
        '''
        Creates a YouTubeSearchTool instance.

        Returns:
            YouTubeSearchTool: The YouTubeSearchTool instance.
        '''
        tool = YouTubeSearchTool()
        return tool
    
    def _output_parser(self) -> StructuredOutputParser:
        '''
        Creates a StructuredOutputParser instance for positive and negative sentiment list.

        Returns:
            StructuredOutputParser: The StructuredOutputParser instance.
        '''
        response_schemas = [
            ResponseSchema(
                name="positive_sentiment", 
                description="list of positive sentiment sentences on the mentioned product.",
                type="list"
            ),
            ResponseSchema(
                name="negative_sentiment",
                description="list of negative sentiment sentences on the mentioned product.",
                type="list"
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser

    def _extract_structured_sentiment(self, output_parser, response:str, product:str):
        '''
        Extracts positive and negative sentiment from the sentiment response.

        Args:
            output_parser (StructuredOutputParser): The output parser.
            response (str): The response.
            product (str): The product.

        Returns:
            dict: The structured positive and negative sentiment.
        '''
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=structured_sentiment_prompt,
            input_variables=["response", "product"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self.llm | output_parser
        structured_sentiment = chain.invoke({"response": response, "product": product})
        return structured_sentiment

    def run(self):
        '''
        Runs the YouTubeSentimentExtractor.

        Returns:
            dict: The structured positive and negative sentiment in dictionary format and video link.
            result: {
                "positive":positive_sentiment, 
                "negative":negative_sentiment
            }
            video_link: {
                "link":link,
                "thumbnail":thumbnail
            }
        '''
        ## Define search tool and output parser
        search_tool = self._youtube_search_tool()
        output_parser = self._output_parser()

        video_id_list = []
        video_link = []
        positive_sentiment = []
        negative_sentiment = []

        ## Search for Youtube videos
        product = self.product.replace(",", " ")
        video_list = search_tool.run(f"{product} review,20")
        video_list = ast.literal_eval(video_list)

        for video in tqdm(video_list):
            video_id = self._extract_videoID(video)
            thumbnail = f"http://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            link = f"https://www.youtube.com/watch?v={video_id}"
            if video_id not in video_id_list:
                try:
                    ## Extract transcript and sentiment
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=('en', 'English'))
                    data = [t['text'] for t in transcript]
                    full_transcript = ' '.join(data)

                    ## Extract sentiment
                    response = self.llm.invoke(get_extract_sentiment_prompt(product, full_transcript))

                    ## Extract structured positive and negative sentiment
                    structured_sentiment = self._extract_structured_sentiment(output_parser, response, product)
                    for sent in structured_sentiment['positive_sentiment']:
                        positive_sentiment.append(sent)
                    for sent in structured_sentiment['negative_sentiment']:
                        negative_sentiment.append(sent)
                    
                    ## Append video ID and link to video_link
                    video_id_list.append(video_id)
                    video_link.append({"link":link, "thumbnail":thumbnail})

                except Exception as e:
                    print(e)
                    continue

        ## Return result and video link
        result = {"positive":positive_sentiment, 
                "negative":negative_sentiment}
        
        return result, video_link
                
            

            
