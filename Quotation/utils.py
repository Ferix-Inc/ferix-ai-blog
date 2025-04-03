from typing import Optional, List, Dict
import json
from pathlib import Path
from schema import get_project_quotation_tool


def generate_quotation(client, prompt: str, files: Optional[List[str]] = None) -> str:
    """
    Calls an API to generate quotation.
    The result is returned as a JSON string that adheres to the Project schema.

    Args:
        client: The Amazon Bedrock client.
        prompt: The text prompt to guide the creation of the quotation.
        files: A list of file paths to parse into the quotation.

    Returns:
        A string of JSON data that follows the Project schema.
    """
    content = [{"text": prompt}]
    if files:
        content += get_documents_list(files)
    messages = [{"role": "user", "content": content}]
    response = client.converse(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        toolConfig={
            "tools": [get_project_quotation_tool()],
            "toolChoice": {"tool": {"name": "project_quotation"}},
        },
    )
    tool_input = response["output"]["message"]["content"][0]["toolUse"]["input"]
    return json.dumps(tool_input, indent=2, ensure_ascii=False)


def text_to_embedding(client, text: str) -> list[float]:
    """
    Converts text to embedding vectors.
    Titan V2 has an 8192-character limit, so the text is truncated if needed.

    Args:
        client: The Amazon Bedrock client used for generating embeddings.
        text: The input text to convert into embeddings.

    Returns:
        A list of float values representing the text's embedding.
    """
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text[:8192]}),
    )

    embedding = json.loads(response["body"].read())["embedding"]
    return embedding


def get_documents_list(files: List[str]) -> List[Dict]:
    documents = []
    for i, file in enumerate(files):
        file_format = Path(file).suffix
        with open(file, "rb") as f:
            document = {
                "document": {
                    "format": file_format[1:],
                    "name": f"document_{i}",
                    "source": {"bytes": f.read()},
                },
            }
        documents.append(document)
    return documents