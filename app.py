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


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
)

css = """
body {
    background: linear-gradient(145deg, #f8fafc 0%, #eef2ff 50%, #faf5ff 100%);
}

#career-ai-app {
    max-width: 1080px;
    margin: 0 auto;
    padding: 24px 18px 30px;
}

#career-hero {
    border: 1px solid rgba(99, 102, 241, 0.16);
    border-radius: 24px;
    background: linear-gradient(135deg, #172554 0%, #3730a3 55%, #6d28d9 100%);
    box-shadow: 0 18px 45px rgba(49, 46, 129, 0.18);
    color: white;
    margin-bottom: 16px;
    overflow: hidden;
    padding: 34px 38px;
}

#career-hero .eyebrow {
    color: #c7d2fe;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin: 0 0 10px;
    text-transform: uppercase;
}

#career-hero h1 {
    color: white;
    font-size: clamp(2rem, 5vw, 3.25rem);
    line-height: 1.05;
    margin: 0;
}

#career-hero .subtitle {
    color: #e0e7ff;
    font-size: 1.05rem;
    margin: 12px 0 5px;
}

#career-hero .intro {
    color: #f8fafc;
    margin: 18px 0 0;
    max-width: 720px;
}

#career-chat {
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    overflow: hidden;
}

#career-input textarea {
    font-size: 1rem;
}

#career-footer {
    color: #64748b;
    font-size: 0.88rem;
    padding: 8px 6px 0;
    text-align: center;
}

#career-footer a {
    color: #4f46e5;
    font-weight: 600;
}

@media (max-width: 640px) {
    #career-ai-app {
        padding: 12px 8px 20px;
    }

    #career-hero {
        border-radius: 18px;
        padding: 26px 22px;
    }
}
"""

with gr.Blocks(
    title=f"{PROFILE_NAME} | Career AI Assistant",
    elem_id="career-ai-app",
) as demo:
    gr.HTML(
        f"""
        <section id="career-hero">
            <p class="eyebrow">Career AI Assistant</p>
            <h1>{PROFILE_NAME}</h1>
            <p class="subtitle">CRM · Revenue Operations · Marketing Automation</p>
            <p class="intro">
                Ask about this professional profile in English or Spanish.<br>
                Pregunta sobre este perfil profesional en inglés o español.
            </p>
        </section>
        """
    )

    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
            height=510,
            layout="bubble",
            placeholder=(
                "Ask a question to explore this professional profile.\n"
                "Haz una pregunta para conocer este perfil profesional."
            ),
            elem_id="career-chat",
        ),
        textbox=gr.Textbox(
            placeholder="Write your question / Escribe tu pregunta",
            container=False,
            elem_id="career-input",
        ),
        examples=[
            "What is your professional background?",
            "¿Cuáles son tus principales habilidades?",
            "What experience do you have with CRM platforms?",
            "¿Cómo puedo contactarte?",
        ],
        example_labels=[
            "Professional background",
            "Habilidades principales",
            "CRM experience",
            "Contacto",
        ],
        title=None,
        description=None,
        autofocus=True,
        flagging_mode="never",
    )

    gr.HTML(
        f"""
        <footer id="career-footer">
            Answers are based on verified professional documents ·
            Respuestas basadas en documentos profesionales verificados ·
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </footer>
        """
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=css,
    )
