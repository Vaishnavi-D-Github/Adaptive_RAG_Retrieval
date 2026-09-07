from pprint import pprint

from adaptive.feature_extractor import (
    extract_features,
)

from adaptive.k_policy import (
    BootstrapKPolicy,
)


TEST_QUERIES = [

    "What is endpoint detection and response?",

    "How do federal agencies prepare for cybersecurity incidents?",

    "Compare the cybersecurity incident response requirements described in the two reports.",

    "What policies are required for event logging?",

    "Summarize the main cybersecurity incident response findings.",

]


def main():

    policy = BootstrapKPolicy()

    print("=" * 80)
    print("ADAPTIVE K PIPELINE - COMPONENT TEST")
    print("=" * 80)


    for query in TEST_QUERIES:

        print()
        print("-" * 80)
        print("QUERY")
        print(query)
        print()


        features = extract_features(
            query
        )

        plan = policy.predict(
            features
        )


        print("FEATURES")
        pprint(
            features.__dict__
        )

        print()

        print("RETRIEVAL PLAN")
        pprint(
            plan.__dict__
        )


if __name__ == "__main__":
    main()