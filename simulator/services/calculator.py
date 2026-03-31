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

    monthly_payment = 0
    total_payable = data["purchaseAmount"]

    if data["paymentType"] == "loan":
        total_payable = data["purchaseAmount"] + (
            data["purchaseAmount"]
            * (data["interestRate"] / 100)
            * (data["loanDuration"] / 12)
        )
        monthly_payment = total_payable / data["loanDuration"]

    new_disposable_income = disposable_income - monthly_payment

    savings_after_purchase = data["currentSavings"]

    if data["paymentType"] == "full":
        savings_after_purchase -= data["purchaseAmount"]

    emergency_buffer = baseline_expenses * 3
    income_ratio = new_disposable_income / data["monthlyIncome"]

    risk_multiplier = 1.0

    if data["incomeStability"] == "unpredictable":
        risk_multiplier += 0.2
    if data["riskTolerance"] == "safe":
        risk_multiplier += 0.1

    adjusted_ratio = income_ratio * risk_multiplier

    if adjusted_ratio > 0.30 and savings_after_purchase > emergency_buffer:
        risk_level = "SAFE"
    elif adjusted_ratio > 0.10:
        risk_level = "TIGHT"
    else:
        risk_level = "RISKY"

    recovery_months = 0
    if data["paymentType"] == "full" and new_disposable_income > 0:
        recovery_months = data["purchaseAmount"] / new_disposable_income

    return {
        "fixed_expenses": round(fixed_expenses, 2),
        "baseline_disposable_income": round(disposable_income, 2),
        "monthly_payment": round(monthly_payment, 2),
        "new_disposable_income": round(new_disposable_income, 2),
        "savings_after_purchase": round(savings_after_purchase, 2),
        "emergency_buffer": round(emergency_buffer, 2),
        "risk_level": risk_level,
        "recovery_months": round(recovery_months, 2),
    }
