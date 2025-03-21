import os
import zipfile
import tempfile
import requests
from gq.pipe import AsyncResult
import time
import asyncio
import logging
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG, get_nvidia
from .prompt import COSMOS_PROMPT

logger = logging.getLogger(__name__)

class CosmosVideoResult(AsyncResult):
    def __init__(self, request_id: str, invoke_url: str, fetch_url_format: str, headers: dict):
        self.request_id = request_id
        self.invoke_url = invoke_url
        self.fetch_url_format = fetch_url_format
        self.headers = headers
        self.session = requests.Session()
        self.result = None
        self.key = f"cosmos-{request_id}"
        self.status = "pending"
        
    async def _poll_result(self, timeout: int = 300) -> dict:
        """Poll for the result of the Cosmos video generation."""
        start_time = time.time()
        fetch_url = self.fetch_url_format + self.request_id
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(fetch_url, headers=self.headers)
                
                if response.status_code == 200:
                    # Check if the content is a zip file
                    if response.headers.get('Content-Type') == 'application/zip':
                        # Save the zip file to a temporary location
                        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                            temp_file.write(response.content)
                            temp_path = temp_file.name
                        
                        # Extract the video file from the zip
                        video_path = self._extract_video_from_zip(temp_path)
                        
                        if video_path:
                            # Here you would typically upload the video to a storage service
                            # and return the URL. For now, we'll just return the local path.
                            self.result = {"file": video_path, "status": "completed"}
                            self.status = "completed"
                            return self.result
                    
                    # If we're still waiting for the result
                    if response.json().get('status') == 'PROCESSING':
                        await asyncio.sleep(5)  # Wait before polling again
                    else:
                        # Process completed
                        self.result = {"file": "placeholder_url", "status": "completed"}
                        self.status = "completed"
                        return self.result
                
                elif response.status_code == 202:
                    # Still processing
                    await asyncio.sleep(5)  # Wait before polling again
                
                else:
                    logger.error(f"Error polling result: {response.status_code} - {response.text}")
                    self.status = "failed"
                    raise Exception(f"Failed to get result: {response.status_code} - {response.text}")
            
            except Exception as e:
                logger.error(f"Error polling result: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
        
        self.status = "timeout"
        raise TimeoutError(f"Timeout waiting for Cosmos video generation result after {timeout} seconds")

    def _extract_video_from_zip(self, zip_path: str) -> Optional[str]:
        """Extract the video file from the zip archive."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create a directory to extract files
                extract_dir = tempfile.mkdtemp()
                zip_ref.extractall(extract_dir)
                
                # Look for video files
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.endswith(('.mp4', '.webm', '.mov')):
                            return os.path.join(root, file)
                
                logger.error(f"No video file found in the zip archive. Contents: {zip_ref.namelist()}")
                return None
        except Exception as e:
            logger.error(f"Error extracting video from zip: {str(e)}")
            return None
        finally:
            # Clean up the zip file
            try:
                os.remove(zip_path)
            except:
                pass

    def get_result(self, timeout: int = 300) -> dict:
        """Get the result of the Cosmos video generation."""
        if self.result:
            return self.result
        
        # Run the polling in a synchronous manner
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._poll_result(timeout))


async def enhance_prompt_with_llm(llm: BaseChatModel, prompt: str) -> str:
    """
    Use an LLM to enhance a simple prompt into a detailed scene description for Cosmos.
    
    Args:
        llm: The language model to use for enhancement
        prompt: The original prompt to enhance
        
    Returns:
        str: The enhanced prompt
    """
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=COSMOS_PROMPT),
                HumanMessage(content=prompt),
            ],
            EMPTY_CALLBACKS_CONFIG,
        )
        enhanced_prompt = response.content
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        # Fall back to the original prompt if enhancement fails
        return prompt


async def create_cosmos_video_async(prompt: str, api_key: Optional[str] = None) -> CosmosVideoResult:
    """
    Asynchronous version of create_cosmos_video that enhances the prompt with an LLM.
    """
    # Enhance the prompt using an LLM
    try:
        enhanced_prompt = await enhance_prompt_with_llm(get_nvidia(), prompt)
    except Exception as e:
        logger.error(f"Error with Nvidia LLM, trying fallback: {str(e)}")
        try:
            # Fallback to another LLM if Nvidia fails
            enhanced_prompt = await enhance_prompt_with_llm(get_llm("Claude 3"), prompt)
        except Exception as e2:
            logger.error(f"Error with fallback LLM: {str(e2)}")
            enhanced_prompt = prompt  # Use original prompt if all LLMs fail
    
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not provided and not found in environment variables")
    
    invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-predict1-7b"
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    # Prepare the command with the enhanced prompt
    command = f'text2world --prompt="{enhanced_prompt}"'
    
    payload = {
        "inputs": [
            {
                "name": "command",
                "shape": [1],
                "datatype": "BYTES",
                "data": [command]
            }
        ],
        "outputs": [
            {
                "name": "status",
                "datatype": "BYTES",
                "shape": [1]
            }
        ]
    }
    
    # Start the session and make the initial request
    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 202:
        # Request accepted, get the request ID
        request_id = response.headers.get("NVCF-REQID")
        if not request_id:
            raise ValueError("No request ID returned from Nvidia Cosmos API")
        
        # Return an async result object
        return CosmosVideoResult(request_id, invoke_url, fetch_url_format, headers)
    else:
        # Handle error
        raise Exception(f"Failed to start Cosmos video generation: {response.status_code} - {response.text}")


def create_cosmos_video(prompt: str, api_key: Optional[str] = None) -> AsyncResult:
    """
    Create a video using Nvidia's Cosmos API with an enhanced prompt.
    
    Args:
        prompt: The text prompt to generate the video from
        api_key: Optional API key, will use environment variable if not provided
    
    Returns:
        AsyncResult: An async result object that can be used to get the video URL
    """
    # Create an async task to handle the prompt enhancement and API call
    async def _create_video():
        try:
            result = await create_cosmos_video_async(prompt, api_key)
            return result
        except Exception as e:
            logger.error(f"Error creating Cosmos video: {str(e)}")
            raise
    
    # Return an AsyncResult that will eventually contain the CosmosVideoResult
    return AsyncResult(_create_video())
