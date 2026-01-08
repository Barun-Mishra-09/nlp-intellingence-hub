
def main():
    analyzer = NLPAnalyzer()

    text = "Elon Musk announced a new SpaceX mission to Mars in Texas."

    categories = ["technology", "business", "sports", "politics"]

    print("\n--- SENTIMENT ---")
    print(analyzer.analyze_sentiment(text))

    print("\n--- ENTITIES ---")
    print(analyzer.extract_entities(text))

    print("\n--- ZERO-SHOT CLASSIFICATION ---")
    print(analyzer.classify_zero_shot(text, categories))

    print("\n--- SUMMARY ---")
    print(analyzer.summarize(text))

if __name__ == "__main__":
    main()


