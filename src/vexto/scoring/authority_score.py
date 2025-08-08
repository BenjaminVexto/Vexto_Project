# authority_score.py

def calculate_authority_score(fetcher_data: dict) -> int:
    """
    Beregner en alternativ 'Authority Score v1' baseret på gratis fetcher-data.
    Maks score: 65 point
    """

    score = 0

    # --- OpenPageRank page_rank_integer ---
    page_rank = fetcher_data.get("page_authority")  # OpenPageRank's 'page_rank_integer'
    if isinstance(page_rank, (int, float)):
        if page_rank >= 6:
            score += 25
        elif page_rank >= 4:
            score += 15
        elif page_rank >= 2:
            score += 5
        # ellers 0

    # --- Global Rank (trafikniveau) ---
    global_rank = fetcher_data.get("global_rank")
    if isinstance(global_rank, int):
        if global_rank < 500_000:
            score += 10
        elif global_rank < 1_000_000:
            score += 5

    # --- Google Business Reviews ---
    gmb_reviews = fetcher_data.get("gmb_review_count")
    if isinstance(gmb_reviews, int):
        if gmb_reviews > 500:
            score += 10
        elif gmb_reviews > 100:
            score += 5

    # --- Gennemsnitlig GMB-rating ---
    gmb_rating = fetcher_data.get("gmb_average_rating")
    if isinstance(gmb_rating, (int, float)):
        if gmb_rating >= 4.3:
            score += 10
        elif gmb_rating >= 3.8:
            score += 5

    # --- Trust-signaler fundet (e-mærket, SSL, GDPR mv.) ---
    trust_signals = fetcher_data.get("trust_signals_found", [])
    if isinstance(trust_signals, list):
        score += min(10, len(trust_signals) * 5)  # 5 point per signal, maks 10

    return min(score, 65)
