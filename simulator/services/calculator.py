def calculate_financial_impact(data: dict):

    fixed_expenses = (
        data["rent"]
        + data["utilities"]
        + data["subscriptionsInsurance"]
    )

    baseline_expenses = (
        fixed_expenses
        + data["existingLoans"]
        + data["variableExpenses"]
    )

    disposable_income = data["monthlyIncome"] - baseline_expenses

    # --- Loan / full-payment calculation (standard amortization) ----------
    monthly_payment = 0
    total_payable = data["purchaseAmount"]

    if data["paymentType"] == "loan":
        principal = data["purchaseAmount"]
        annual_rate = data["interestRate"]
        months = data["loanDuration"]

        if annual_rate > 0:
            monthly_rate = annual_rate / 100 / 12
            # Standard amortization: M = P * r(1+r)^n / ((1+r)^n - 1)
            factor = (1 + monthly_rate) ** months
            monthly_payment = principal * (monthly_rate * factor) / (factor - 1)
            total_payable = monthly_payment * months
        else:
            # 0 % interest – simple division
            monthly_payment = principal / months
            total_payable = principal

    new_disposable_income = disposable_income - monthly_payment

    # --- Savings impact ---------------------------------------------------
    savings_after_purchase = data["currentSavings"]
    if data["paymentType"] == "full":
        savings_after_purchase -= data["purchaseAmount"]

    # Emergency buffer: 3 months of ALL obligations including new payment
    emergency_buffer = (baseline_expenses + monthly_payment) * 3

    # --- Risk scoring -----------------------------------------------------
    if data["monthlyIncome"] > 0:
        income_ratio = new_disposable_income / data["monthlyIncome"]
    else:
        income_ratio = -1.0  # no income → maximum risk

    # Higher risk_multiplier = stricter (used as divisor)
    risk_multiplier = 1.0

    # Income stability
    if data["incomeStability"] == "unpredictable":
        risk_multiplier += 0.20
    elif data["incomeStability"] == "sometimes_changes":
        risk_multiplier += 0.10

    # Risk tolerance
    if data["riskTolerance"] == "safety":
        risk_multiplier += 0.10
    elif data["riskTolerance"] == "balanced":
        risk_multiplier += 0.05

    # Dependents
    dependents = data.get("dependents", 0)
    risk_multiplier += dependents * 0.05

    # Household responsibility
    household = data.get("householdResponsibilityLevel", "not_applicable")
    if household == "all_or_most":
        risk_multiplier += 0.15
    elif household == "half":
        risk_multiplier += 0.10
    elif household == "small_part":
        risk_multiplier += 0.05

    # DIVIDE so that higher risk → lower adjusted ratio → harder to be SAFE
    adjusted_ratio = income_ratio / risk_multiplier

    # --- Risk level determination -----------------------------------------
    if savings_after_purchase < 0:
        risk_level = "RISKY"
    elif adjusted_ratio > 0.30 and savings_after_purchase > emergency_buffer:
        risk_level = "SAFE"
    elif adjusted_ratio > 0.10 and savings_after_purchase > 0:
        risk_level = "TIGHT"
    else:
        risk_level = "RISKY"

    # --- Recovery months --------------------------------------------------
    recovery_months = 0
    if data["paymentType"] == "full":
        if new_disposable_income > 0:
            recovery_months = data["purchaseAmount"] / new_disposable_income
        else:
            recovery_months = -1  # cannot recover with current income

    return {
        "fixed_expenses": round(fixed_expenses, 2),
        "baseline_expenses": round(baseline_expenses, 2),
        "baseline_disposable_income": round(disposable_income, 2),
        "monthly_payment": round(monthly_payment, 2),
        "total_payable": round(total_payable, 2),
        "new_disposable_income": round(new_disposable_income, 2),
        "savings_after_purchase": round(savings_after_purchase, 2),
        "emergency_buffer": round(emergency_buffer, 2),
        "risk_level": risk_level,
        "recovery_months": round(recovery_months, 2),
    }
