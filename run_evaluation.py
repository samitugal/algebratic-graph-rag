#!/usr/bin/env python3
"""
RAG Evaluation System Runner
Quick script to run AI-as-a-Judge evaluation
"""

import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.main_evaluation import main, quick_test, custom_evaluation


def show_menu():
    print("\n" + "=" * 60)
    print("🚀 RAG EVALUATION SYSTEM")
    print("=" * 60)
    print("1. 📊 Full Comprehensive Evaluation (5 questions × 5 runs = CLT)")
    print("2. ⚡ Quick Test (1 question × 3 runs)")
    print("3. 🎯 Custom Evaluation (3 custom questions × 5 runs)")
    print("4. ❓ Help & Information")
    print("5. 🚪 Exit")
    print("=" * 60)


def show_help():
    print("\n📋 EVALUATION SYSTEM HELP")
    print("-" * 40)
    print("Bu sistem AI-as-a-Judge yöntemi kullanarak RAG sistemlerini değerlendirir.")
    print("\n🔍 Değerlendirilen Metrikler:")
    print("  • Faithfulness: Kaynak dokümanlarla tutarlılık")
    print("  • Answer Relevancy: Soruyla ilgi düzeyi")
    print("  • Context Precision: Bağlam kalitesi")
    print("  • Completeness: Yanıt tamlığı")
    print("  • Hallucination: Uydurma bilgi tespiti")
    print("\n🔬 Karşılaştırılan Yöntemler:")
    print("  • PageRank GraphRAG")
    print("  • KNN GraphRAG")
    print("  • Basic GraphRAG")
    print("\n📊 Sonuçlar JSON formatında kaydedilir ve detaylı rapor üretilir.")
    print("\n🧮 Multiple Runs System (Central Limit Theorem):")
    print("  • Her method multiple kez çalıştırılır")
    print("  • CLT ile AI judge variance azaltılır")
    print("  • Statistical olarak güvenilir sonuçlar elde edilir")
    print("\n⚠️  Not: İlk çalıştırmadan önce:")
    print("  1. Neo4j veritabanının çalıştığından emin olun")
    print("  2. Graf verilerinin oluşturulduğunu kontrol edin")
    print("  3. API key'lerinizi .env dosyasında ayarlayın")


def main_menu():
    while True:
        show_menu()

        try:
            choice = input("\nSeçiminizi yapın (1-5): ").strip()

            if choice == "1":
                print("\n🚀 Starting Full Comprehensive Evaluation...")
                main()

            elif choice == "2":
                print("\n⚡ Starting Quick Test...")
                quick_test()

            elif choice == "3":
                print("\n🎯 Starting Custom Evaluation...")
                custom_evaluation()

            elif choice == "4":
                show_help()

            elif choice == "5":
                print("\n👋 Evaluation sistemi kapatılıyor...")
                break

            else:
                print("\n❌ Geçersiz seçim! Lütfen 1-5 arasında bir sayı girin.")

        except KeyboardInterrupt:
            print("\n\n⏹️  İşlem kullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            print(f"\n❌ Hata oluştu: {str(e)}")
            print("Lütfen tekrar deneyin veya sistem yöneticisine başvurun.")


if __name__ == "__main__":
    main_menu()
