from rest_framework import serializers


class OptionalFloatField(serializers.FloatField):
    def to_internal_value(self, data):
        if data in ("", None):
            return None
        return super().to_internal_value(data)


class OptionalDateField(serializers.DateField):
    def to_internal_value(self, value):
        if value in ("", None):
            return None
        return super().to_internal_value(value)


class FinancialSimulationSerializer(serializers.Serializer):

    monthlyIncome = serializers.FloatField(min_value=0)

    rent = serializers.FloatField(min_value=0)
    utilities = serializers.FloatField(min_value=0)
    subscriptionsInsurance = serializers.FloatField(min_value=0)

    existingLoans = serializers.FloatField(min_value=0)
    variableExpenses = serializers.FloatField(min_value=0)
    currentSavings = serializers.FloatField(min_value=0)
    dependents = serializers.IntegerField(min_value=0)

    HOUSEHOLD_CHOICES = [
        ("all_or_most", "All or Most of It"),
        ("half", "About Half"),
        ("not_applicable", "Not Applicable"),
    ]

    incomeStability_CHOICES = [
        ("very_stable", "Very Stable"),
        ("mostly_stable", "Mostly Stable"),
        ("unpredictable", "Unpredictable"),
    ]

    riskTolerance_CHOICES = [
        ("safe", "I Prefer Safety"),
        ("balanced", "Balanced"),
        ("risk_ok", "I'm Okay With Risk"),
    ]

    householdResponsibilityLevel = serializers.ChoiceField(choices=HOUSEHOLD_CHOICES)
    incomeStability = serializers.ChoiceField(choices=incomeStability_CHOICES)
    riskTolerance = serializers.ChoiceField(choices=riskTolerance_CHOICES)

    purchaseAmount = serializers.FloatField(min_value=0)
    paymentType = serializers.ChoiceField(choices=[("full", "Full"), ("loan", "Loan")])

    loanDuration = serializers.IntegerField(required=False, min_value=1)
    interestRate = serializers.FloatField(required=False, min_value=0)

    # Optional future goal plan inputs (used by AI guidance when provided)
    planName = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        max_length=120,
    )
    targetAmount = OptionalFloatField(
        required=False,
        allow_null=True,
        min_value=0,
    )
    targetDate = OptionalDateField(
        required=False,
        allow_null=True,
        input_formats=["%d/%m/%Y", "%Y-%m-%d"],
    )
    goalDescription = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        max_length=500,
    )

    def validate(self, data):
        if data["paymentType"] == "loan":
            if "loanDuration" not in data or "interestRate" not in data:
                raise serializers.ValidationError(
                    "Loan duration and interest rate required for loan."
                )
        return data
