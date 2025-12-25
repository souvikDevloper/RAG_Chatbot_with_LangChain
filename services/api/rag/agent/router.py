def route_intent(plan_intent: str) -> str:
    # already validated in Plan, but keep a guard
    if plan_intent in {"qa","summarize","compare","quote","troubleshoot"}:
        return plan_intent
    return "qa"
