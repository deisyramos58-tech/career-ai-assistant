import json
import os
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv(override=True)

client = OpenAI()

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "me"

PROFILE_NAME = os.getenv("PROFILE_NAME", "Alex Morgan")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "alex@example.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_notification(message):
    """Send an optional Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"sent": False, "reason": "Telegram is not configured."}

    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=10,
    )
    response.raise_for_status()
    return {"sent": True}


def read_document(file_path):
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    return ""


def load_career_context():
    sections = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        text = read_document(file_path).strip()
        if text:
            sections.append(f"SOURCE: {file_path.name}\n{text}")

    if not sections:
        raise ValueError("No PDF or TXT source documents were found in me/.")

    return "\n\n".join(sections)


career_context = load_career_context()


def record_user_details(email, name=None, notes=None):
    notification = send_notification(
        "New contact from Career AI\n"
        f"Name: {name or 'Not provided'}\n"
        f"Email: {email}\n"
        f"Notes: {notes or 'No additional information'}"
    )
    return {"recorded": True, "notification": notification}


def record_unknown_question(question):
    notification = send_notification(
        f"Unanswered Career AI question:\n{question}"
    )
    return {"recorded": True, "notification": notification}


tools = [
    {
        "type": "function",
        "name": "record_user_details",
        "description": (
            "Use only when a user voluntarily provides an email address "
            f"and asks to be contacted by {PROFILE_NAME}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "name": {"type": ["string", "null"]},
                "notes": {"type": ["string", "null"]},
            },
            "required": ["email", "name", "notes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "record_unknown_question",
        "description": (
            "Use when the source documents do not contain enough information "
            "to answer a career-related question reliably."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def handle_tool_calls(response):
    tool_outputs = []
    available_tools = {
        "record_user_details": record_user_details,
        "record_unknown_question": record_unknown_question,
    }

    for item in response.output:
        if item.type != "function_call":
            continue

        function = available_tools.get(item.name)
        arguments = json.loads(item.arguments)
        result = (
            function(**arguments)
            if function
            else {"recorded": False, "error": f"Unknown tool: {item.name}"}
        )
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result),
            }
        )

    return tool_outputs


system_prompt = f"""
You are the professional Career AI Assistant for {PROFILE_NAME}.

- Respond in the same language used by the user.
- Base every factual claim only on the source documents below.
- Do not invent employers, dates, metrics, certifications, projects, degrees,
  responsibilities, or technical expertise.
- Clearly distinguish professional experience from skills currently being
  studied or developed.
- Politely decline unrelated requests.
- Never reveal API keys, tokens, environment variables, internal tool
  definitions, or implementation details.
- If the documents cannot answer a career-related question, call
  record_unknown_question, explain that the information is unavailable, and
  offer the contact email {CONTACT_EMAIL}.
- If a visitor voluntarily provides an email and asks to be contacted, call
  record_user_details. Never invent or demand contact information.

SOURCE DOCUMENTS

{career_context}
"""


def chat(message, history):
    messages = []

    for item in history:
        if isinstance(item, dict):
            messages.append(
                {"role": item["role"], "content": item["content"]}
            )

    messages.append({"role": "user", "content": message})

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=messages,
        tools=tools,
    )

    for _ in range(3):
        tool_outputs = handle_tool_calls(response)
        if not tool_outputs:
            return response.output_text

        messages.extend(response.output)
        messages.extend(tool_outputs)
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=messages,
            tools=tools,
        )

    return response.output_text


demo = gr.ChatInterface(
    fn=chat,
    title=f"{PROFILE_NAME} | Career AI Assistant",
    description="Ask about this sample professional profile in English or Spanish.",
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
