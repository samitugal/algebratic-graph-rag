import json
import time
from scripts.evaluation_pipeline import AcademicEvaluationPipeline
from statistics import mean, stdev


def test_evaluator_consistency():
    """Test AI judge consistency with same question multiple runs"""

    print("🧪 Testing AI Judge Consistency")
    print("=" * 50)

    pipeline = AcademicEvaluationPipeline()

    # Single test question
    test_question = "What is object-oriented programming?"

    print(f"❓ Testing Question: {test_question}")
    print(f"🔄 Running 3 times to check consistency...")

    results = []

    for run in range(3):
        print(f"\n📋 Run {run + 1}/3")

        # Compare all methods
        comparison = pipeline.compare_methods(test_question)

        if comparison:
            run_scores = {}
            for result in comparison.results:
                run_scores[result.method_name] = result.overall_score
                print(f"   {result.method_name}: {result.overall_score:.3f}")

            results.append(run_scores)
            print(f"   🏆 Winner: {comparison.best_method}")

        # Small delay between runs
        time.sleep(2)

    # Analyze consistency
    if len(results) >= 2:
        print(f"\n📊 CONSISTENCY ANALYSIS")
        print("-" * 30)

        for method in results[0].keys():
            scores = [r[method] for r in results]
            avg_score = mean(scores)
            std_dev = stdev(scores) if len(scores) > 1 else 0
            variance_pct = (std_dev / avg_score * 100) if avg_score > 0 else 0

            print(f"{method}:")
            print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
            print(f"  Average: {avg_score:.3f} ± {std_dev:.3f}")
            print(f"  Variance: {variance_pct:.1f}%")

            if variance_pct > 10:
                print(f"  ⚠️  HIGH VARIANCE (>{variance_pct:.1f}%)")
            elif variance_pct > 5:
                print(f"  ⚡ Moderate variance ({variance_pct:.1f}%)")
            else:
                print(f"  ✅ Low variance ({variance_pct:.1f}%)")
            print()

        # Overall consistency assessment
        all_variances = []
        for method in results[0].keys():
            scores = [r[method] for r in results]
            if len(scores) > 1:
                std_dev = stdev(scores)
                avg_score = mean(scores)
                variance_pct = (std_dev / avg_score * 100) if avg_score > 0 else 0
                all_variances.append(variance_pct)

        avg_variance = mean(all_variances) if all_variances else 0

        print(f"📈 OVERALL ASSESSMENT:")
        print(f"Average variance across all methods: {avg_variance:.1f}%")

        if avg_variance > 15:
            print("🚨 CRITICAL: AI Judge highly inconsistent!")
            print("   Recommendation: Multiple evaluation runs needed")
        elif avg_variance > 8:
            print("⚠️  WARNING: Moderate inconsistency detected")
            print("   Recommendation: Consider ensemble evaluation")
        else:
            print("✅ GOOD: AI Judge relatively consistent")
            print("   Single evaluation runs should be reliable")


if __name__ == "__main__":
    test_evaluator_consistency()
