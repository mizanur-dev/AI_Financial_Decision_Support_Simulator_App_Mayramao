from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from django.conf import settings

import hashlib
import json
import re
from collections import OrderedDict

# ── Module-level LLM singleton (created once, reused across requests) ────
_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=1500,
        )
    return _llm_instance


# ── Simple bounded LRU cache for identical requests ─────────────────────
_CACHE_MAX = 128
_cache: OrderedDict[str, dict] = OrderedDict()


def _cache_key(user_data: dict, calculation: dict) -> str:
    relevant = {
        "monthlyIncome": user_data.get("monthlyIncome"),
        "rent": user_data.get("rent"),
        "utilities": user_data.get("utilities"),
        "subscriptionsInsurance": user_data.get("subscriptionsInsurance"),
        "existingLoans": user_data.get("existingLoans"),
        "variableExpenses": user_data.get("variableExpenses"),
        "currentSavings": user_data.get("currentSavings"),
        "dependents": user_data.get("dependents"),
        "householdResponsibilityLevel": user_data.get("householdResponsibilityLevel"),
        "incomeStability": user_data.get("incomeStability"),
        "riskTolerance": user_data.get("riskTolerance"),
        "purchaseAmount": user_data.get("purchaseAmount"),
        "paymentType": user_data.get("paymentType"),
        "loanDuration": user_data.get("loanDuration"),
        "interestRate": user_data.get("interestRate"),
        "planName": user_data.get("planName"),
        "targetAmount": user_data.get("targetAmount"),
        "targetDate": str(user_data.get("targetDate")),
        "goalDescription": user_data.get("goalDescription"),
        "risk_level": calculation.get("risk_level"),
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


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

    lines = ["\nFuture Goal Plan:"]
    if plan_name not in (None, ""):
        lines.append(f"- Plan Name: {plan_name}")
    if target_amount is not None:
        lines.append(f"- Target Amount: {target_amount}")
    if target_date is not None:
        lines.append(f"- Target Date: {target_date}")
    if short_description not in (None, ""):
        lines.append(f"- Short Description: {short_description}")

    return "\n".join(lines) + "\n"


def _build_purchase_section(user_data: dict, calculation: dict) -> str:
    lines = [
        f"- Purchase Amount: {user_data.get('purchaseAmount')}",
        f"- Payment Type: {user_data.get('paymentType')}",
    ]
    if user_data.get("paymentType") == "loan":
        lines.append(f"- Loan Duration: {user_data.get('loanDuration')} months")
        lines.append(f"- Annual Interest Rate: {user_data.get('interestRate')}%")
        total_interest = round(
            calculation.get("total_payable", 0) - user_data.get("purchaseAmount", 0), 2
        )
        lines.append(f"- Total Interest Cost: {total_interest}")
    lines.append(f"- Monthly Payment: {calculation.get('monthly_payment')}")
    lines.append(f"- Total Payable: {calculation.get('total_payable')}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    candidate = match.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def generate_ai_guidance(user_data: dict, calculation: dict):

    # Check cache first — identical inputs return instantly
    key = _cache_key(user_data, calculation)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    llm = _get_llm()

    goal_plan_section = _build_goal_plan_section(user_data)
    purchase_section = _build_purchase_section(user_data, calculation)

    recovery_val = calculation.get("recovery_months", 0)
    if recovery_val == -1:
        recovery_desc = "Cannot recover (negative disposable income)"
    elif recovery_val == 0 and user_data.get("paymentType") == "loan":
        recovery_desc = f"N/A (loan repayment over {user_data.get('loanDuration', '?')} months)"
    else:
        recovery_desc = f"{recovery_val} months"

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
- Base ALL advice strictly on the actual numbers below. Do NOT assume or invent figures.

User Profile:
- Monthly Income: {monthlyIncome}
- Dependents: {dependents}
- Income Stability: {incomeStability}
- Risk Tolerance: {riskTolerance}
- Household Responsibility: {householdResponsibilityLevel}

Monthly Expenses:
- Rent: {rent}
- Utilities: {utilities}
- Subscriptions / Insurance: {subscriptionsInsurance}
- Existing Loan Payments: {existingLoans}
- Variable Expenses: {variableExpenses}
- Total Fixed Expenses: {fixed_expenses}
- Total Baseline Expenses: {baseline_expenses}

Current Financial Position:
- Current Savings: {currentSavings}
- Baseline Disposable Income (before purchase): {baseline_disposable_income}

Purchase Details:
{purchase_section}

Post-Purchase Impact:
- New Disposable Income: {new_disposable_income}
- Savings After Purchase: {savings_after_purchase}
- Emergency Buffer Needed (3 months of obligations): {emergency_buffer}
- Recovery Timeline: {recovery_desc}
- Overall Risk Level: {risk_level}
{goal_plan_section}""")

    chain = prompt | llm

    template_vars = {
        **user_data,
        **calculation,
        "goal_plan_section": goal_plan_section,
        "purchase_section": purchase_section,
        "recovery_desc": recovery_desc,
    }
    template_vars.setdefault("householdResponsibilityLevel", "not_applicable")

    response = chain.invoke(template_vars)
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
    result = {
        "assessment_title": _assessment_title_from_risk_level(risk_level),
        "risk_level": risk_level,
        "guidance": guidance.strip() if isinstance(guidance, str) else "",
        "key_insights": normalized_insights,
        "safer_alternatives": normalized_alternatives,
    }

    # Store in cache
    _cache[key] = result
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)

    return result
