# Copyright (c) Microsoft & Workshop Contributors. All rights reserved.
"""
Personal Career Copilot - Multi-Agent Workflow Powered by Local Ollama.
Lab 02: Sequential 4-agent pipeline running 100% offline on your local machine with tool calling.
"""

import json
import logging
import os
import sys
import time
import urllib.parse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ollama-multi-agent")

# --- Agent Prompts ---

RESUME_PARSER_INSTRUCTIONS = """\
You are the Resume Parser and Content Router.
Your input contains a resume and usually a job description - BOTH must be preserved.

TASK 1 - Parse the resume into a structured candidate profile.
TASK 2 - Copy the job description verbatim into the pass-through section below.

Output EXACTLY these two labeled sections:

[PARSED RESUME]
1) Candidate Profile & Role
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
- The [JOB DESCRIPTION PASS-THROUGH] section MUST contain the FULL, UNMODIFIED JD text.
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
- Extract requirements ONLY from [JOB DESCRIPTION PASS-THROUGH].
- Copy [PARSED RESUME] verbatim.
- Keep required vs preferred clearly separated.
"""

MATCHING_AGENT_INSTRUCTIONS = """\
You are the Matching Agent.
Your input is the Job Description Analyst output. It contains two clearly labeled sections:
  - [JD REQUIREMENTS] - the structured job requirements; use this as the target profile.
  - [PARSED RESUME PASS-THROUGH] - the candidate's parsed profile; use this as the candidate profile.

Compare [PARSED RESUME PASS-THROUGH] vs [JD REQUIREMENTS] and produce an evidence-based fit report.

Scoring (100 total):
- Required Skills: 40 points
- Experience: 25 points
- Certifications: 15 points
- Preferred Skills: 10 points
- Domain Alignment: 10 points

Output exactly these sections:
1) Fit Score: <Score>/100 (with clear breakdown math)
2) Matched Skills: (bullet list of direct matches)
3) Missing Skills: (bullet list of missing required skills)
4) Partially Matched Skills: (skills with partial overlap)
5) Experience Alignment: (summary of years and relevance)
6) Certification Gaps: (missing credentials)
7) Overall Assessment: (short executive summary of candidate readiness)

Rules:
- Be objective, strict, and evidence-based.
- Missing Skills feeds directly into the roadmap planner.
"""

GAP_ANALYZER_INSTRUCTIONS = """\
You are the Gap Analyzer and Roadmap Planner.
Create a practical upskilling plan from the matching report.

Tool Usage (Required):
- For major missing skills or certification gaps, invoke the `search_learning_resources` tool.
- Integrate the returned learning links into the Suggested Resources section for each gap card.

Output format:
1) Personalized Upskilling Roadmap for Target Role
2) Detailed Gap Breakdown Cards (produce a card for EVERY identified gap):
   - Skill / Area:
   - Priority: (High / Medium / Low)
   - Current Level:
   - Target Level:
   - Suggested Resources: (include curated links returned by tool)
   - Estimated Time:
   - Practical Quick-Win Project:
3) Recommended Learning Order (numbered sequence)
4) Timeline Summary (week-by-week plan)
5) Motivational Closing & Interview Tip
"""

# --- Tools ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_learning_resources",
            "description": "Searches for official documentation, Microsoft Learn, and tutorials for a given technology or skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "The technical skill, framework, or cloud tool (e.g. 'Kubernetes', 'Terraform', 'Azure AI').",
                    },
                },
                "required": ["skill"],
            },
        },
    }
]


def search_learning_resources(skill: str) -> str:
    clean_skill = skill.strip()
    encoded = urllib.parse.quote(clean_skill)
    return (
        f"1. Microsoft Learn: https://learn.microsoft.com/search/?terms={encoded}\n"
        f"2. Official Documentation: https://devdocs.io/#q={encoded}\n"
        f"3. GitHub Community Guides: https://github.com/topics/{encoded.lower()}"
    )


TOOL_MAP = {
    "search_learning_resources": search_learning_resources,
}


def create_ollama_client() -> tuple[OpenAI, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
    )
    return client, model


def execute_agent_step(client: OpenAI, model: str, agent_name: str, system_prompt: str, user_content: str) -> str:
    logger.info(f"▶️ [{agent_name}] running on Local Ollama ({model})...")
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ [{agent_name}] finished in {elapsed:.2f}s")
    return response.choices[0].message.content or ""


def execute_gap_analyzer(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    logger.info(f"▶️ [GapAnalyzer] running with tool-calling on Local Ollama ({model})...")
    start_time = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    max_turns = 4
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        response_msg = response.choices[0].message

        if response_msg.tool_calls:
            messages.append(response_msg)
            for tc in response_msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    skill = args.get("skill", "Technology")
                except Exception:
                    skill = "Technology"

                logger.info(f"🔧 Tool Call (turn {turn+1}): {func_name}(skill='{skill}')")
                func = TOOL_MAP.get(func_name, search_learning_resources)
                tool_output = func(skill)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })
        else:
            elapsed = time.time() - start_time
            logger.info(f"✅ [GapAnalyzer] completed with tool synthesis in {elapsed:.2f}s")
            return response_msg.content or "Roadmap generated successfully."

    elapsed = time.time() - start_time
    logger.info(f"✅ [GapAnalyzer] finished in {elapsed:.2f}s")
    return messages[-1].get("content") or "Roadmap generation complete."


def run_career_copilot(resume_text: str, jd_text: str) -> dict:
    client, model = create_ollama_client()
    overall_start = time.time()

    print("\n" + "=" * 70)
    print(f" 💻 Personal Career Copilot: 4-Agent Workflow on Local Ollama ({model})")
    print("=" * 70)

    combined_input = f"--- CANDIDATE RESUME ---\n{resume_text}\n\n--- TARGET JOB DESCRIPTION ---\n{jd_text}"

    print("\n[Step 1/4] Parsing candidate resume...")
    parsed_resume_output = execute_agent_step(
        client, model, "ResumeParser", RESUME_PARSER_INSTRUCTIONS, combined_input
    )

    print("\n[Step 2/4] Analyzing job requirements...")
    jd_analysis_output = execute_agent_step(
        client, model, "JobDescriptionAgent", JOB_DESCRIPTION_INSTRUCTIONS, parsed_resume_output
    )

    print("\n[Step 3/4] Calculating fit score & skill alignment...")
    matching_output = execute_agent_step(
        client, model, "MatchingAgent", MATCHING_AGENT_INSTRUCTIONS, jd_analysis_output
    )

    print("\n[Step 4/4] Generating personalized roadmap with resource lookup...")
    final_roadmap = execute_gap_analyzer(
        client, model, GAP_ANALYZER_INSTRUCTIONS, matching_output
    )

    total_time = time.time() - overall_start
    print(f"\n✨ Workflow completed on Local Ollama in {total_time:.2f} seconds total!")

    return {
        "parsed_resume": parsed_resume_output,
        "jd_analysis": jd_analysis_output,
        "matching_report": matching_output,
        "final_roadmap": final_roadmap,
        "total_time_seconds": total_time,
    }


def main():
    sample_resume = """
Jane Doe - Senior Software Engineer
Summary: 5+ years building backend web applications and distributed systems in Python, Django, and AWS.
Experience:
- Senior Backend Engineer at CloudScale (3 years): Architected RESTful microservices handling 10k req/sec.
- Software Engineer at DataCorp (2 years): Built automated data pipelines with PostgreSQL and Redis.
Certifications: AWS Certified Solutions Architect - Associate.
Education: B.S. in Computer Science.
"""

    sample_jd = """
Senior AI Cloud Solutions Engineer - Contoso AI
Requirements:
- 5+ years software engineering experience with Python.
- Hands-on experience deploying Kubernetes clusters and container workloads.
- Infrastructure as Code with Terraform and Azure Resource Manager.
- Familiarity with Azure AI Foundry / Agent frameworks and Large Language Model fine-tuning.
- Preferred: Azure Solutions Architect Expert certification, Go programming.
"""

    print("=" * 70)
    print(" 🌟 Lab 02 - Multi-Agent Workflow: Local Ollama Runner")
    print("=" * 70)

    try:
        result = run_career_copilot(sample_resume, sample_jd)
    except Exception as e:
        print(f"\n❌ Connection Error: Could not reach Ollama at {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}")
        print("👉 Start Ollama with `ollama serve` and pull a model with `ollama pull qwen2.5:7b`.\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" 📊 INTERMEDIATE AGENT OUTPUT: MATCHING REPORT")
    print("=" * 70)
    print(result["matching_report"])

    print("\n" + "=" * 70)
    print(" 🎓 FINAL AGENT OUTPUT: PERSONALIZED LEARNING ROADMAP")
    print("=" * 70)
    print(result["final_roadmap"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
