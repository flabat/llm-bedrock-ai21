from typing import Optional

import boto3
import llm
import json
from pydantic import Field, field_validator

DEFAULT_JURASSIC_CHAT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

@llm.hookimpl
def register_models(register):
    register(
        BedrockJurassic("ai21.j2-ultra-v1"),
        aliases=("bedrock-jurassic2-ultra", 'bj2u'),
    ),
    register(
        BedrockJurassic("ai21.j2-mid-v1"),
        aliases=("bedrock-jurassic2-mid", 'bj2m'),
    )

class BedrockJurassic(llm.Model):
    can_stream: bool = False

    class Options(llm.Options):
        verbose: bool = Field(
            description="Whether to print verbose output from the model", default=False
        )

        maxTokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=2048,
        )

        temperature: Optional[float] = Field(
            description="Temperature",
            default=0.5,
        )

        topP: Optional[float] = Field(
            description="Top P",
            default=0.5,
        )
        @field_validator("maxTokens")
        def validate_length(cls, maxTokens):
            if not (0 < maxTokens <= 1_000_000):
                raise ValueError("maxTokens must be in range 1-1,000,000")
            return maxTokens

    def __init__(self, model_id):
        self.model_id = model_id
        self.default_system_prompt = None

    def build_jurassic2_chat_prompt(self, prompt, conversation):
        prompt_bits = []
        # First figure out the system prompt
        system_prompt = None
        if prompt.system:
            system_prompt = prompt.system
        else:
            # Look for a system prompt in the conversation
            if conversation is not None:
                for prev_response in conversation.responses:
                    if prev_response.prompt.system:
                        system_prompt = prev_response.prompt.system
        if system_prompt is None:
            system_prompt = (
                self.default_system_prompt or DEFAULT_JURASSIC_CHAT_SYSTEM_PROMPT
            )

        # Now build the prompt pieces
        first = True
        if conversation is not None:
            for prev_response in conversation.responses:
                prompt_bits.append("<s>[INST] ")
                if first:
                    prompt_bits.append(
                        f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
                    )
                first = False
                prompt_bits.append(
                    f"{prev_response.prompt.prompt} [/INST] ",
                )
                prompt_bits.append(
                    f"{prev_response.text()} </s>",
                )

        # Add the latest prompt
        if not prompt_bits:
            # Start with the system prompt
            prompt_bits.append("<s>[INST] ")
            prompt_bits.append(
                f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
            )
        else:
            prompt_bits.append("<s>[INST] ")
        prompt_bits.append(f"{prompt.prompt} [/INST] ")
        return prompt_bits
    

    def execute(self, prompt, stream, response, conversation):
        client = boto3.client('bedrock-runtime')

        prompt_str = "".join(self.build_jurassic2_chat_prompt(prompt, conversation))
        prompt_json = {
            "prompt": prompt_str,
            "maxTokens": prompt.options.maxTokens,
            "temperature": prompt.options.temperature,
            "topP": prompt.options.topP,
        }
        prompt.prompt_json = prompt_json
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=json.dumps(prompt_json)
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                if chunk:
                    response = json.loads(chunk.get("bytes").decode())
                    completion = response["generation"]
                    yield completion

        else:
            bedrock_response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(prompt_json),
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)

            completion = response.response_json["completions"][0]["data"]["text"]
            

            yield completion