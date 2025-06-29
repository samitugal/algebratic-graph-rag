import os
import warnings

warnings.filterwarnings("ignore")

from scripts.evaluation_pipeline import AcademicEvaluationPipeline


def main():
    """Main function to run comprehensive evaluation with Central Limit Theorem"""

    print("🚀 Starting Comprehensive RAG Evaluation System (CLT Multiple Runs)")
    print("=" * 70)

    # Initialize with 5 runs for AI judge variance compensation
    pipeline = AcademicEvaluationPipeline(n_runs=5)

    test_questions = [
        "Explain computer programming and artificial intelligence",
        "What is object-oriented programming?",
        "Tell me about Turkey and Izmir",
        "What is computer programming and artificial intelligence?",
        "Give me the information about Object-Oriented Programming",
    ]

    print(f"📝 Evaluating {len(test_questions)} questions...")
    print(f"🔬 Methods: PageRank GraphRAG, KNN GraphRAG, Basic GraphRAG")
    print(
        f"🧮 Each method × {pipeline.n_runs} runs = {len(test_questions) * 3 * pipeline.n_runs} total evaluations"
    )
    print(f"🎯 Central Limit Theorem applied for AI judge variance reduction")
    print()

    # Run comprehensive evaluation
    try:
        comparisons = pipeline.batch_evaluation(test_questions)

        if not comparisons:
            print("❌ No successful evaluations completed!")
            return

        # Generate detailed report
        report = pipeline.generate_final_report(comparisons)

        # Print summary
        print(f"\n✅ Evaluation completed! Results saved to: {pipeline.results_dir}")

        # Additional insights
        print("\n📈 DETAILED INSIGHTS")
        print("-" * 60)

        # Best performing method overall
        best_methods = [comp.best_method for comp in comparisons]
        method_counts = {}
        for method in best_methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        overall_best = max(method_counts.items(), key=lambda x: x[1])
        print(
            f"🏆 Most frequently best method: {overall_best[0]} ({overall_best[1]}/{len(comparisons)} questions)"
        )

        # Performance by question type
        turkish_questions = [
            q for q in test_questions if any(char in q for char in "çğıöşüÇĞIİÖŞÜ")
        ]
        english_questions = [q for q in test_questions if q not in turkish_questions]

        if turkish_questions:
            print(f"🇹🇷 Turkish questions evaluated: {len(turkish_questions)}")
        if english_questions:
            print(f"🇺🇸 English questions evaluated: {len(english_questions)}")

        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Full report saved to: comprehensive_evaluation_report.json")

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()


def quick_test():
    """Quick test with single question (3 runs for speed)"""

    print("🧪 Running Quick Test (3 runs per method)")
    print("-" * 40)

    pipeline = AcademicEvaluationPipeline(n_runs=3)

    # Single test question
    test_question = "What is artificial intelligence?"

    print(f"❓ Question: {test_question}")

    # Compare all methods
    comparison = pipeline.compare_methods(test_question)

    if comparison:
        print(f"\n🏆 Best method: {comparison.best_method}")

        for result in comparison.results:
            print(f"\n📊 {result.method_name}:")
            print(f"   Overall Score: {result.overall_score:.3f}")
            for metric, score in result.metric_scores.items():
                print(f"   {metric}: {score:.3f}")
    else:
        print("❌ No results obtained")


def custom_evaluation():
    """Custom evaluation with user-defined questions (5 runs per method)"""

    print("🎯 Custom Evaluation Mode (CLT Applied)")
    print("-" * 40)

    pipeline = AcademicEvaluationPipeline(n_runs=5)

    # Custom questions - you can modify these
    custom_questions = [
        "Explain machine learning algorithms",
        "What are the benefits of graph databases?",
        "How does natural language processing work?",
    ]

    print(f"📝 Custom questions to evaluate: {len(custom_questions)}")

    for i, question in enumerate(custom_questions, 1):
        print(f"\n{i}. {question}")

    # Run evaluation
    comparisons = pipeline.batch_evaluation(custom_questions)

    # Generate report
    report = pipeline.generate_final_report(comparisons)
    print(f"\n✅ Custom evaluation completed! Results saved to: {pipeline.results_dir}")


if __name__ == "__main__":
    print("Choose evaluation mode:")
    print("1. Full comprehensive evaluation")
    print("2. Quick test with single question")
    print("3. Custom evaluation")

    try:
        choice = input("\nEnter choice (1-3, default=1): ").strip()

        if choice == "2":
            quick_test()
        elif choice == "3":
            custom_evaluation()
        else:
            main()

    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
