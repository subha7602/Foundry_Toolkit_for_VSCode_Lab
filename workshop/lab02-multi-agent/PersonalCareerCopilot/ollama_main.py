# Copyright (c) Microsoft & Workshop Contributors. All rights reserved.
"""
Personal Career Copilot - Multi-Agent Workflow Powered by Local Ollama.
Supports Text, PDF, and DOCX resumes, VS Code Agent Inspector streaming, and CLI.
"""

import json
import logging
import os
import sys
import uuid
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from openai import OpenAI

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ollama-multi-agent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

RESUME_PARSER_INSTRUCTIONS = """\
You are the Resume Parser and Content Router.
Your input contains a resume and usually a job description - BOTH must be preserved.

TASK 1 - Parse the resume into a structured candidate profile.
TASK 2 - Copy the job description verbatim into the pass-through section below.

Output EXACTLY these two labeled sections:

[PARSED RESUME]
1) Candidate Profile
2) Technical Skills (grouped categories)
3) Soft Skills
4) Certifications & Awards
5) Domain Experience
6) Notable Achievements

[JOB DESCRIPTION PASS-THROUGH]
<Copy the complete job description here exactly as given. Do NOT summarize or paraphrase.
If no job description is present, write only: No job description provided.>

Rules:
- Use only explicit or strongly implied evidence for the resume sections.
- Do not invent skills, titles, or experience.
- Keep resume bullets concise; no long paragraphs.
- The [JOB DESCRIPTION PASS-THROUGH] section MUST contain the FULL, UNMODIFIED JD text.
  Omitting or truncating it breaks the downstream Job Description Agent.
"""

JOB_DESCRIPTION_INSTRUCTIONS = """\
You are the Job Description Analyst and Resume Relay.
Your input is the Resume Parser output. It contains two clearly labeled sections:
  - [PARSED RESUME] - copy this verbatim to [PARSED RESUME PASS-THROUGH] in your output.
  - [JOB DESCRIPTION PASS-THROUGH] - extract the structured JD requirements from here only.

Output EXACTLY these two labeled sections:

[JD REQUIREMENTS]
1) Role Overview
2) Required Skills
3) Preferred Skills
4) Experience Required
5) Certifications Required
6) Education
7) Domain / Industry
8) Key Responsibilities

[PARSED RESUME PASS-THROUGH]
<Copy the complete [PARSED RESUME] section here exactly as given. Do NOT summarize or paraphrase.>

Rules:
- Extract requirements ONLY from [JOB DESCRIPTION PASS-THROUGH] - do not use [PARSED RESUME] content.
- Copy [PARSED RESUME] verbatim - the Matching Agent depends on it downstream.
- Keep required vs preferred clearly separated.
- Only use what the JD states; do not invent hidden requirements.
- Flag vague requirements briefly.
- If the JD pass-through says "No job description provided", write in [JD REQUIREMENTS]:
  "No JD available - cannot extract requirements. Ask the user to re-submit with a job description."
"""

MATCHING_AGENT_INSTRUCTIONS = """\
You are the Matching Agent.
Your input is the Job Description Analyst output. It contains two clearly labeled sections:
  - [JD REQUIREMENTS] - the structured job requirements; use this as the target profile.
  - [PARSED RESUME PASS-THROUGH] - the candidate's parsed profile; use this as the candidate profile.

Compare [PARSED RESUME PASS-THROUGH] vs [JD REQUIREMENTS] and produce an evidence-based fit report.

Scoring (100 total):
- Required Skills 40
- Experience 25
- Certifications 15
- Preferred Skills 10
- Domain Alignment 10

Output exactly these sections:
1) Fit Score (with breakdown math)
2) Matched Skills
3) Missing Skills
4) Partially Matched
5) Experience Alignment
6) Certification Gaps
7) Overall Assessment

Rules:
- Be objective and evidence-only.
- Keep partial vs missing separate.
- Keep Missing Skills precise; it feeds roadmap planning.
"""

GAP_ANALYZER_INSTRUCTIONS = """\
You are the Gap Analyzer and Roadmap Planner.
Create a practical upskilling plan from the matching report.

Tool Usage (Required):
- For EVERY High and Medium priority gap, invoke tool `search_learning_resources`.
- Include curated documentation links in Suggested Resources for each gap card.

Output format:
1) Personalized Learning Roadmap for [Role Title]
2) One DETAILED card per gap (produce ALL cards, not just the first):
   - Skill
   - Priority (High/Medium/Low)
   - Current Level
   - Target Level
   - Suggested Resources (include documentation URLs from tool results)
   - Estimated Time
   - Quick Win Project
3) Recommended Learning Order (numbered list)
4) Timeline Summary (week-by-week)
5) Motivational Note
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_learning_resources",
            "description": "Searches for official documentation and learning resources for a given skill or technology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "The skill or technology to search for"}
                },
                "required": ["skill"],
            },
        },
    }
]


def extract_pdf_text(filepath: str) -> str:
    """Extracts text from a local PDF resume."""
    if not pypdf:
        return "[PDF parsing error: pypdf library is not installed. Run 'pip install pypdf']"
    try:
        reader = pypdf.PdfReader(filepath)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    except Exception as e:
        logger.error(f"Failed to extract PDF {filepath}: {e}")
        return f"[Error reading PDF {filepath}: {e}]"


def extract_docx_text(filepath: str) -> str:
    """Extracts text from a local Word DOCX resume."""
    if not docx:
        return "[DOCX parsing error: python-docx library is not installed. Run 'pip install python-docx']"
    try:
        doc = docx.Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        logger.error(f"Failed to extract DOCX {filepath}: {e}")
        return f"[Error reading DOCX {filepath}: {e}]"


def find_file(path_str: str) -> str:
    """Finds a file if it exists directly or relative to workshop directories."""
    clean = path_str.strip("'\"><(),:;\n\r")
    if os.path.isfile(clean):
        return os.path.abspath(clean)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(os.getcwd(), clean),
        os.path.join(script_dir, clean),
        os.path.join(script_dir, "workshop/lab02-multi-agent/PersonalCareerCopilot", clean),
        os.path.join(os.getcwd(), "Foundry_Toolkit_for_VSCode_Lab/workshop/lab02-multi-agent/PersonalCareerCopilot", clean),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def preprocess_input(user_input: str) -> str:
    """Auto-detects PDF and DOCX file paths in user input and extracts resume content."""
    words = user_input.split()
    for word in words:
        clean = word.strip("'\"><(),:;\n\r")
        lower_name = clean.lower()
        if lower_name.endswith(".pdf") or lower_name.endswith(".docx"):
            resolved = find_file(clean)
            if resolved:
                if lower_name.endswith(".pdf"):
                    logger.info(f"Extracting resume text from PDF: {resolved}")
                    extracted = extract_pdf_text(resolved)
                    user_input = user_input.replace(
                        word, f"\n[RESUME EXTRACTED FROM PDF: {os.path.basename(resolved)}]\n{extracted}\n"
                    )
                elif lower_name.endswith(".docx"):
                    logger.info(f"Extracting resume text from DOCX: {resolved}")
                    extracted = extract_docx_text(resolved)
                    user_input = user_input.replace(
                        word, f"\n[RESUME EXTRACTED FROM DOCX: {os.path.basename(resolved)}]\n{extracted}\n"
                    )
    return user_input


def search_learning_resources(skill: str) -> str:
    encoded = urllib.parse.quote(skill.strip())
    return (
        f"1. Microsoft Learn: https://learn.microsoft.com/search/?terms={encoded}\n"
        f"2. Official Documentation: https://devdocs.io/#q={encoded}\n"
        f"3. GitHub Community: https://github.com/topics/{encoded.lower()}"
    )


def execute_step(client: OpenAI, instructions: str, user_content: str, use_tools: bool = False) -> str:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]

    kwargs = {"model": OLLAMA_MODEL, "messages": messages, "temperature": 0.2}
    if use_tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            skill_arg = args.get("skill", "tech")
            tool_res = search_learning_resources(skill_arg)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_res})

        final_resp = client.chat.completions.create(model=OLLAMA_MODEL, messages=messages, temperature=0.2)
        return final_resp.choices[0].message.content or ""

    return msg.content or ""


def start_server(host="127.0.0.1", port=8088):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self):
            self._send_json({"status": "healthy", "agent": "PersonalCareerCopilot", "provider": "Ollama", "model": OLLAMA_MODEL})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else ""
                try:
                    data = json.loads(body) if body else {}
                except Exception:
                    data = {"input": body}

                prompt = data.get("input") or data.get("message") or body
                if isinstance(prompt, list):
                    prompt = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in prompt])
                elif isinstance(prompt, dict):
                    prompt = prompt.get("text", "") or str(prompt)

                conv_id = data.get("conversation_id", f"conv_{uuid.uuid4().hex[:8]}")
                resp_id = data.get("response_id", f"resp_{uuid.uuid4().hex[:8]}")
                item_id = f"item_{uuid.uuid4().hex[:8]}"

                logger.info(f"Processing prompt with Ollama ({OLLAMA_MODEL}): {str(prompt)[:60]}...")

                # Send SSE headers IMMEDIATELY
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

                def send_event(ev_type: str, ev_data: dict):
                    payload = {"type": ev_type, **ev_data}
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                def send_delta(text: str):
                    send_event("response.text.delta", {"delta": text})

                send_event("response.created", {"response": {"id": resp_id, "conversation_id": conv_id, "status": "in_progress"}})
                send_event("response.output_item.added", {"item": {"id": item_id, "type": "message", "role": "assistant", "content": []}})
                send_event("response.content_part.added", {"part": {"type": "text", "text": ""}})

                client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

                # Auto-Extract Text from PDF / DOCX if supplied
                processed_input = preprocess_input(str(prompt))

                # Step 1: Resume Parser
                send_delta("Running multi-agent analysis with Ollama...\n\n[Step 1/4] Parsing candidate profile...\n")
                parsed_resume = execute_step(client, RESUME_PARSER_INSTRUCTIONS, processed_input)

                # Step 2: Job Description Agent
                send_delta("[Step 2/4] Analyzing job requirements...\n")
                jd_analysis = execute_step(client, JOB_DESCRIPTION_INSTRUCTIONS, parsed_resume)

                # Step 3: Matching Agent
                send_delta("[Step 3/4] Evaluating fit score and skills gap...\n")
                matching_report = execute_step(client, MATCHING_AGENT_INSTRUCTIONS, jd_analysis)

                # Step 4: Gap Analyzer
                send_delta("[Step 4/4] Building personalized learning roadmap with resources...\n\n")
                final_roadmap = execute_step(client, GAP_ANALYZER_INSTRUCTIONS, matching_report, use_tools=True)

                full_output = f"=== MATCHING REPORT ===\n{matching_report}\n\n=== UPSKILLING ROADMAP ===\n{final_roadmap}"
                send_delta(f"\n{full_output}")

                send_event("response.output_item.done", {"item": {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": full_output}]}})
                send_event("response.done", {"response": {"id": resp_id, "conversation_id": conv_id, "status": "completed", "output": [{"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": full_output}]}]}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            except Exception as e:
                logger.error(f"Error handling request: {e}")
                try:
                    self.wfile.write(f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass

        def log_message(self, format, *args):
            pass

    print("=" * 70)
    print(f"Ollama Multi-Agent Server running on http://{host}:{port}")
    print(f"Model: {OLLAMA_MODEL}")
    print("Features: Text + PDF + DOCX Extraction | Live SSE Streaming")
    print("=" * 70 + "\n")

    httpd = HTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        sample = """
Resume: sample_resume.pdf
Job Description: Senior AI Cloud Engineer, requires Kubernetes, Terraform, Azure AI Foundry, LLM fine-tuning.
"""
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        r1 = execute_step(client, RESUME_PARSER_INSTRUCTIONS, preprocess_input(sample))
        r2 = execute_step(client, JOB_DESCRIPTION_INSTRUCTIONS, r1)
        r3 = execute_step(client, MATCHING_AGENT_INSTRUCTIONS, r2)
        r4 = execute_step(client, GAP_ANALYZER_INSTRUCTIONS, r3, use_tools=True)
        print(f"=== MATCHING REPORT ===\n{r3}\n\n=== UPSKILLING ROADMAP ===\n{r4}")
    else:
        start_server()
