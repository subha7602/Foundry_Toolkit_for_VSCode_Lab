# Copyright (c) Microsoft & Workshop Contributors. All rights reserved.
"""
Personal Career Copilot - Multi-Agent Workflow Powered by OpenRouter.
Self-contained 4-agent sequential workflow runner compatible with VS Code Agent Inspector and CLI.
"""

import json
import logging
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("openrouter-multi-agent")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

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

    kwargs = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.2}
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

        final_resp = client.chat.completions.create(model=OPENROUTER_MODEL, messages=messages, temperature=0.2)
        return final_resp.choices[0].message.content or ""

    return msg.content or ""


def run_career_copilot(input_text: str) -> str:
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set in your .env file."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/microsoft-foundry/workshop",
            "X-Title": "AI Agents Workshop",
        },
    )

    logger.info("[Step 1/4] Running ResumeParser on OpenRouter...")
    parsed_resume = execute_step(client, RESUME_PARSER_INSTRUCTIONS, input_text)

    logger.info("[Step 2/4] Running JobDescriptionAgent on OpenRouter...")
    jd_analysis = execute_step(client, JOB_DESCRIPTION_INSTRUCTIONS, parsed_resume)

    logger.info("[Step 3/4] Running MatchingAgent on OpenRouter...")
    matching_report = execute_step(client, MATCHING_AGENT_INSTRUCTIONS, jd_analysis)

    logger.info("[Step 4/4] Running GapAnalyzer with tool-calling on OpenRouter...")
    final_roadmap = execute_step(client, GAP_ANALYZER_INSTRUCTIONS, matching_report, use_tools=True)

    return f"=== MATCHING REPORT ===\n{matching_report}\n\n=== UPSKILLING ROADMAP ===\n{final_roadmap}"


def start_server(host="127.0.0.1", port=8088):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._send_json({"status": "healthy", "agent": "PersonalCareerCopilot", "model": OPENROUTER_MODEL})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                data = json.loads(body) if body else {}
                prompt = data.get("input") or data.get("message") or body
                if isinstance(prompt, list):
                    prompt = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in prompt])
            except Exception:
                prompt = body

            logger.info(f"Processing multi-agent pipeline with OpenRouter ({OPENROUTER_MODEL})...")
            output = run_career_copilot(prompt)

            self._send_json({
                "status": "completed",
                "response": output,
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "text", "text": output}]}],
            })

        def log_message(self, format, *args):
            pass

    print("=" * 70)
    print(f"OpenRouter Multi-Agent Server running on http://{host}:{port}")
    print(f"Model: {OPENROUTER_MODEL}")
    print("Compatible with VS Code Command: Foundry Toolkit: Open Agent Inspector")
    print("=" * 70 + "\n")

    httpd = HTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        sample = """
Resume: Jane Doe, 5 years Python backend developer with AWS and PostgreSQL.
Job Description: Senior AI Cloud Engineer, requires Kubernetes, Terraform, Azure AI Foundry, LLM fine-tuning.
"""
        print("\n--- CLI Test Output ---\n")
        print(run_career_copilot(sample))
    else:
        start_server()
