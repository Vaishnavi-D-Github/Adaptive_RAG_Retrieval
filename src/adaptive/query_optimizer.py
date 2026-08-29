from .schemas import QueryFeatures


def normalize_query(
    query: str,
) -> str:

    return " ".join(
        str(query).strip().split()
    )


def optimize_query(
    query: str,
    features: QueryFeatures,
) -> str:

    query = normalize_query(
        query
    )

    additions = []

    if features.is_comparison:

        additions.extend(
            [
                "comparison",
                "differences",
                "similarities",
            ]
        )


    if features.is_procedural:

        additions.extend(
            [
                "procedure",
                "steps",
                "requirements",
            ]
        )


    if features.is_policy:

        additions.extend(
            [
                "policy",
                "requirements",
                "rules",
            ]
        )


    if not additions:

        return query


    return (
        query
        + " "
        + " ".join(
            dict.fromkeys(
                additions
            )
        )
    )