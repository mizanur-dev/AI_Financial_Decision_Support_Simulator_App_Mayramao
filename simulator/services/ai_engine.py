from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from django.conf import settings

import json
import re


def _assessment_title_from_risk_level(risk_level: str) -> str:
    risk_level = (risk_level or "").upper()
    if risk_level == "SAFE":
        return "You're in Good Shape!"
    if risk_level == "TIGHT":
        return "Proceed with Caution"
    return "High Financial Risk"


def _build_goal_plan_section(user_data: dict) -> str:
    plan_name = user_data.get("planName")
    target_amount = user_data.get("targetAmount")
    target_date = user_data.get("targetDate")
    short_description = user_data.get("goalDescription")

    has_any = any(
        v not in (None, "")
        for v in (plan_name, target_amount, target_date, short_description)
    )
    if not has_any:
        return ""

    lines = ["Future Goal Plan (optional):"]
    if plan_name not in (None, ""):
        lines.append(f"- Plan Name: {plan_name}")
    if target_amount is not None:
        lines.append(f"- Target Amount: {target_amount}")
    if target_date is not None:
        # date object -> ISO-like string
        lines.append(f"- Target Date: {target_date}")
    if short_description not in (None, ""):
        lines.append(f"- Short Description: {short_description}")

    return "\n".join(lines) + "\n"


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    # Try to pull the first JSON object from the model output.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    candidate = match.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def generate_ai_guidance(user_data: dict, calculation: dict):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )

    goal_plan_section = _build_goal_plan_section(user_data)

    prompt = ChatPromptTemplate.from_template("""
You are a conservative financial advisor.

Return ONLY valid JSON (no markdown, no extra text) with this schema:
{{
    "guidance": string,
    "key_insights": [
        {{"title": string, "detail": string}}
    ],
    "safer_alternatives": [string]
}}

Rules:
- Keep guidance concise and practical (1-2 short paragraphs).
- key_insights: 2-4 items, each with a short title + 1 sentence detail.
- safer_alternatives: 3-6 items as short actionable bullets.
- If risk is SAFE, alternatives can be "optional optimizations".
- If a Future Goal Plan is provided, incorporate it (impact on goal timeline and adjustments).

User Profile:
Income: {monthlyIncome}
dependents: {dependents}
Income Stability: {incomeStability}
Risk Tolerance: {riskTolerance}

{goal_plan_section}

Financial Results:
Risk Level: {risk_level}
Disposable Income After Purchase: {new_disposable_income}
Savings After Purchase: {savings_after_purchase}
Emergency Buffer Needed: {emergency_buffer}
Monthly Payment: {monthly_payment}
Recovery Months (if full payment): {recovery_months}
""")

    chain = prompt | llm

    response = chain.invoke({**user_data, **calculation, "goal_plan_section": goal_plan_section})
    raw_text = getattr(response, "content", "")

    parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        parsed = {}

    guidance = parsed.get("guidance") if isinstance(parsed.get("guidance"), str) else raw_text

    key_insights = parsed.get("key_insights")
    if not isinstance(key_insights, list):
        key_insights = []
    normalized_insights: list[dict] = []
    for item in key_insights:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        detail = item.get("detail")
        if isinstance(title, str) and isinstance(detail, str) and title.strip() and detail.strip():
            normalized_insights.append({"title": title.strip(), "detail": detail.strip()})

    safer_alternatives = parsed.get("safer_alternatives")
    if not isinstance(safer_alternatives, list):
        safer_alternatives = []
    normalized_alternatives: list[str] = []
    for alt in safer_alternatives:
        if isinstance(alt, str) and alt.strip():
            normalized_alternatives.append(alt.strip())

    risk_level = calculation.get("risk_level")
    return {
        "assessment_title": _assessment_title_from_risk_level(risk_level),
        "risk_level": risk_level,
        "guidance": guidance.strip() if isinstance(guidance, str) else "",
        "key_insights": normalized_insights,
        "safer_alternatives": normalized_alternatives,
    }
