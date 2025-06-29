# 📊 Academic Evaluation Framework for PageRank-GraphRAG

## 🎯 Overview

Bu proje, **PageRank-weighted Hierarchical GraphRAG** sistemlerini değerlendirmek için kapsamlı bir akademik çerçeve sunar. AI-as-a-Judge metodolojisi kullanarak objective ve reproducible sonuçlar üretir.

## 🔬 Bilimsel Katkılar

### Yenilikçi Yaklaşımlar:
1. **PageRank-Weighted Path Traversal**: Geleneksel vector similarity'ye ek olarak graph path weights'i kullanır
2. **Hierarchical Graph Construction**: Multi-level summarization ile graph depth artırır  
3. **Hybrid Seed Selection**: KNN + semantic similarity ile optimal başlangıç noktaları
4. **Path Reliability Scoring**: `Score = Semantic_Similarity × Path_Weight`

### Literatürdeki Konumu:
- **Microsoft GraphRAG**: Community detection only
- **LightRAG**: Keyword + local search only
- **PathRAG**: Flow-based path retrieval
- **🆕 Bizim Yaklaşım**: PageRank + Hierarchical + Path-weighted scoring

## 📁 Evaluation Directory Structure

Her evaluation çalıştırıldığında şu yapı oluşturulur:

```
evaluation_results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── 📋 experiment_metadata.json     # Experiment configuration
    ├── 📄 README.md                    # Experiment documentation
    ├── 📊 executive_summary.json       # Key findings summary
    │
    ├── raw_data/                       # Raw evaluation data
    │   └── comparison_*.json           # Individual comparisons
    │
    ├── processed_data/                 # Processed analysis data
    │   └── detailed_results.csv        # All results in CSV format
    │
    ├── statistical_analysis/           # Statistical tests & significance
    │   ├── aggregate_statistics.json   # Overall statistics
    │   └── statistical_tests.json      # Significance test data
    │
    ├── tables/                         # Academic paper ready tables
    │   ├── method_comparison.csv       # Method comparison matrix
    │   ├── method_summary.csv          # Summary statistics
    │   └── latex_method_comparison.tex # LaTeX table for papers
    │
    ├── visualizations/                 # Publication-quality plots
    │   ├── score_distributions.png     # Score histograms
    │   ├── method_comparison_boxplot.png
    │   └── win_rates.png               # Performance pie chart
    │
    └── logs/                           # Execution logs
```

## 🚀 Quick Start

### 1. Basit Kullanım
```python
from scripts.evaluation_pipeline import AcademicEvaluationPipeline

# Initialize pipeline
pipeline = AcademicEvaluationPipeline()

# Test questions
questions = [
    "Explain artificial intelligence and machine learning",
    "What is object-oriented programming?",
    "Yapay zeka ve makine öğrenmesi nedir?"
]

# Run evaluation (creates full academic report)
comparisons = pipeline.batch_evaluation(questions)

# Results automatically saved to evaluation_results/
print(f"Results saved to: {pipeline.results_dir}")
```

### 2. Terminal'den Kullanım
```bash
# Interactive evaluation menu
python run_evaluation.py

# Direct evaluation
python scripts/main_evaluation.py
```

## 📊 Evaluation Metrics

### AI-as-a-Judge Metrics:
1. **Faithfulness** (0.0-1.0): Source document consistency
2. **Answer Relevancy** (0.0-1.0): Question relevance  
3. **Context Precision** (0.0-1.0): Retrieved context quality
4. **Completeness** (0.0-1.0): Answer comprehensiveness
5. **Hallucination Detection** (0.0-1.0): False information detection

### Method Comparison:
- **PageRank GraphRAG**: Path-weighted traversal + hierarchical context
- **KNN GraphRAG**: K-nearest neighbor similarity search
- **Basic GraphRAG**: Simple graph neighborhood retrieval

## 📈 Academic Output Files

### For Publications:

#### 1. LaTeX Table (Ready to use)
```latex
% tables/latex_method_comparison.tex
\begin{table}[h]
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
Method & Mean Score & Std Dev & Wins & Win Rate (\%) \\
\hline
PageRank GraphRAG & 0.847 & 0.123 & 8 & 80.0 \\
KNN GraphRAG & 0.756 & 0.098 & 2 & 20.0 \\
Basic GraphRAG & 0.623 & 0.156 & 0 & 0.0 \\
\hline
\end{tabular}
\caption{Method Performance Comparison}
\label{tab:method_comparison}
\end{table}
```

#### 2. Statistical Data (CSV format)
- **processed_data/detailed_results.csv**: All numerical results
- **tables/method_summary.csv**: Summary statistics  
- **statistical_analysis/aggregate_statistics.json**: Full statistics

#### 3. Publication-Quality Plots
- High-resolution (300 DPI) PNG files
- Seaborn-styled academic plots
- Ready for journal submission

## 🔬 Research Applications

### Ideal for Academic Papers on:
- **Information Retrieval**: Novel graph-based retrieval methods
- **Question Answering**: Multi-hop reasoning evaluation
- **Knowledge Graphs**: Hierarchical summarization techniques
- **AI Evaluation**: AI-as-a-Judge methodologies

### Conference/Journal Targets:
- **ACL 2025**: NLP and language understanding
- **EMNLP 2025**: Empirical methods  
- **ICLR 2026**: Learning representations
- **Information Retrieval Journal**: IR methods
- **IEEE TKDE**: Knowledge and data engineering

## 🧪 Example Research Questions

1. **How does path-weighted PageRank improve multi-hop reasoning?**
2. **What is the impact of hierarchical summarization on context quality?**  
3. **How do different graph traversal strategies affect answer relevance?**
4. **Can hybrid seed selection reduce hallucination in GraphRAG?**

## 📋 Reproducibility Checklist

✅ **Data**: All raw evaluation data saved (JSON format)  
✅ **Code**: Complete pipeline available  
✅ **Config**: Experiment metadata recorded  
✅ **Stats**: Statistical significance data  
✅ **Plots**: Publication-ready visualizations  
✅ **Tables**: LaTeX-formatted results  

## 🔧 Advanced Configuration

### Custom Evaluation Setup:
```python
# Custom experiment with specific name
pipeline = AcademicEvaluationPipeline()
pipeline.setup_experiment_directory("my_paper_experiment")

# Custom questions and methods
custom_questions = ["Your specific domain questions..."]
custom_methods = ["PageRank_GraphRAG", "KNN_GraphRAG"]

results = pipeline.batch_evaluation(custom_questions, custom_methods)
```

### Statistical Analysis Extensions:
```python
# Add scipy for advanced statistical tests
from scipy import stats

# T-test between methods
method1_scores = [...]  # From detailed_results.csv
method2_scores = [...]
t_stat, p_value = stats.ttest_ind(method1_scores, method2_scores)
```

## 📊 Sample Results Interpretation

### Win Rate Analysis:
```json
{
  "PageRank_GraphRAG": {"wins": 8, "win_rate": 80.0},
  "KNN_GraphRAG": {"wins": 2, "win_rate": 20.0},
  "Basic_GraphRAG": {"wins": 0, "win_rate": 0.0}
}
```

### Statistical Significance:
- **Sample size**: Each method evaluated on N questions
- **Effect size**: Mean score differences
- **Variance**: Standard deviation across questions
- **Confidence**: 95% confidence intervals available

## 🎯 Best Practices for Academic Use

### 1. Experiment Design:
- Use diverse question sets (min 20-30 questions)
- Include both English and Turkish for multilingual evaluation  
- Balance question complexity levels
- Document domain-specific adaptations

### 2. Statistical Rigor:
- Run multiple trials for statistical significance
- Report confidence intervals
- Use appropriate statistical tests (t-test, ANOVA)
- Address multiple comparisons problem

### 3. Reporting:
- Include full methodology description
- Report both individual metrics and overall scores
- Provide error bars in plots
- Make data and code available for reproducibility

## 🔍 Troubleshooting

### Common Issues:
1. **No visualization files**: Install matplotlib/seaborn
2. **Empty results**: Check Neo4j connection and graph data
3. **API errors**: Verify OpenAI API keys in .env file
4. **Memory issues**: Reduce batch size for large question sets

### Debug Mode:
```python
pipeline = AcademicEvaluationPipeline()
# Add verbose logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Citation

When using this evaluation framework in academic work:

```bibtex
@software{pagerank_graphrag_eval,
  title={PageRank-Weighted Hierarchical GraphRAG Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/algebratic-graph-rag}
}
```

## 🤝 Contributing

For research collaborations and improvements:
1. Fork the repository
2. Create feature branch for your experiments
3. Document methodological changes
4. Submit pull request with evaluation results

---

**🎓 Ready for Academic Publication**: This framework provides all necessary components for rigorous academic evaluation of GraphRAG systems, with emphasis on reproducibility and statistical significance. 