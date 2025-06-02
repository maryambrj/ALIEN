import instructor
import json
from pydantic import BaseModel
from typing import Type
from openai import AsyncOpenAI
import google.generativeai as genai


class ExtractionClient:
    """ExtractionClient is a wrapper for OpenAI and Gemini APIs to extract structured data from unstructured text.
    It uses the instructor library to handle the API calls and responses.
    Args:
        provider (str): The provider to use for extraction. Can be "openai" or "gemini".
        api_key (str): The API key for the provider.
        model_name (str): The model name to use for extraction.
        temperature (float): The temperature for the model. Default is 0.0.
        max_tokens (int): The maximum number of tokens to generate. Default is 1024.
        max_retries (int): The maximum number of retries for the API call. Default is 10.
    Example:
        client = ExtractionClient(
            provider="openai",
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 10
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client, self.mode = self._configure_client()

    def _configure_client(self):
        """Configures the client based on the provider and returns the client and mode."""
        if self.provider == "openai":
            client = AsyncOpenAI(api_key=self.api_key)
            wrapped_client = instructor.patch(
                client, mode=instructor.Mode.FUNCTIONS)
            wrapped_client.max_retries = self.max_retries
            return wrapped_client, "openai"

        elif self.provider == "gemini":
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(model_name=self.model_name)
            wrapped_client = instructor.from_gemini(
                model, mode=instructor.Mode.GEMINI_JSON,
                use_async=True
            )
            wrapped_client.max_retries = self.max_retries
            return wrapped_client, "gemini"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def extract(
        self, prompt: str, schema: Type[BaseModel],
        system_prompt: str = ""
    ) -> BaseModel:
        """Extracts structured data from unstructured text using the 
        configured client.
        Args:
            prompt (str): The prompt to send to the model.
            schema (Type[BaseModel]): The schema to use for the response.
            system_prompt (str): The system prompt to send to the model. 
            Default is "".
        Returns:
            BaseModel: The structured data extracted from the unstructured text.
        """
        if self.mode == "openai":
            response = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt !=
                        "" else "You are a helpful assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                response_model=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )  # type: ignore
            # type: ignore
            return json.loads(response.model_dump_json())
        elif self.mode == "gemini":
            response = await self.client.messages.create(  # type: ignore
                messages=[
                    {"role": "system", "content": system_prompt if system_prompt !=
                        "" else "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_model=schema,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            # type: ignore
            return json.loads(response.model_dump_json())

        else:
            raise ValueError("Client mode not configured correctly.")
