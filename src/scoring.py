def privacy_score(flags):
    score = 100
    if flags.get("DATA_SHARING"): score -= 30
    if flags.get("TRACKING"): score -= 25
    if flags.get("THIRD_PARTY"): score -= 20
    if not flags.get("USER_RIGHTS"): score -= 15
    return max(score, 0)
