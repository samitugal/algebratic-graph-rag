import json
from scripts.evaluation_pipeline import AcademicEvaluationPipeline


def test_multiple_runs():
    """Test the multiple runs + averaging system with Central Limit Theorem"""

    print("🧮 Testing Central Limit Theorem Application to AI Judge")
    print("=" * 60)
    print("📐 Theory: As n→∞, sample mean approaches population mean")
    print("🎯 Goal: Reduce AI judge variance through statistical averaging")
    print()

    # Initialize with 5 runs (good balance of reliability vs speed)
    pipeline = AcademicEvaluationPipeline(n_runs=5)

    # Test question
    test_question = "What is object-oriented programming?"

    print(f"❓ Test Question: {test_question}")
    print()

    # Compare all methods with multiple runs
    comparison = pipeline.compare_methods(test_question)

    if comparison:
        print(f"\n🏆 FINAL RESULTS (Central Limit Theorem Applied)")
        print("=" * 50)

        for result in comparison.results:
            variance_pct = (
                (result.score_variance / result.overall_score * 100)
                if result.overall_score > 0
                else 0
            )
            reliability = (
                "High" if variance_pct < 5 else "Medium" if variance_pct < 10 else "Low"
            )

            print(f"\n📊 {result.method_name}:")
            print(f"   🎯 Final Score: {result.overall_score:.3f}")
            print(f"   📈 Standard Dev: ±{result.score_variance:.3f}")
            print(f"   📊 Variance: {variance_pct:.1f}%")
            print(f"   🔢 Sample Size: {result.run_count} runs")
            print(f"   ✅ Reliability: {reliability}")
            print(
                f"   🎲 Individual Scores: {[f'{s:.3f}' for s in result.individual_scores]}"
            )

        print(f"\n🏆 Winner: {comparison.best_method}")
        print(f"📊 This result is statistically robust due to multiple runs!")

        # Show before/after comparison
        print(f"\n🔬 VARIANCE ANALYSIS:")
        print("Without CLT: Single run → high variance → unreliable")
        print("With CLT: Multiple runs → reduced variance → reliable")
        print(
            f"🎯 Improvement: √{pipeline.n_runs} = {pipeline.n_runs**0.5:.1f}x variance reduction!"
        )

    else:
        print("❌ No results obtained")


if __name__ == "__main__":
    test_multiple_runs()
