import requests
import json
import os
from loguru import logger


class OllamaLLM:

    def __init__(self, model="mistral"):
        self.model = model

        # ✅ Works for both LOCAL and DOCKER
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        logger.info(f"Ollama URL: {self.base_url}")
        logger.info(f"Model: {self.model}")

    # ==========================
    # NORMAL GENERATION
    # ==========================
    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 512
                    }
                },
                timeout=300
            )

            response.raise_for_status()
            data = response.json()

            return data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generate error: {e}")
            return "⚠️ LLM service unavailable."

    # ==========================
    # STREAMING GENERATION (FIXED)
    # ==========================
    def stream_generate(self, prompt: str):
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 512
                    }
                },
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            buffer = ""

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")

                    buffer += token

                    # ✅ Flush readable chunks
                    if token.endswith((" ", ".", "\n", ",", ":", ";")):
                        yield buffer
                        buffer = ""

                    # ✅ Final flush
                    if data.get("done", False):
                        if buffer:
                            yield buffer
                        break

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama streaming error: {e}")
            yield "⚠️ Streaming failed. Check Ollama connection."
