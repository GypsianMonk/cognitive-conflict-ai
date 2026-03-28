"""
Cognitive Conflict AI — CLI Entry Point
Usage: python main.py --query "Your question here"
"""

import argparse
import json
import yaml
from pathlib import Path

from engine.debate_manager import DebateManager
from scoring.metrics import record_output


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Default config
    return {
        "model": {"provider": "ollama", "name": "mistral", "temperature": 0.7},
        "agents": {"debate_rounds": 3, "enable_expert_injection": True},
        "scoring": {"relevance_weight": 0.4, "coherence_weight": 0.4, "contradiction_penalty": 0.2},
        "memory": {"enabled": True},
    }


def print_output(output, verbose: bool = False):
    print("\n" + "=" * 60)
    print("🧠 COGNITIVE CONFLICT AI — RESULT")
    print("=" * 60)
    print(f"📋 Query     : {output.query}")
    print(f"🌐 Domain    : {output.domain or 'General'}")
    print(f"⚙️  Complexity : {output.complexity}")
    print(f"🔄 Rounds    : {output.total_debate_rounds}")
    print(f"🤝 Converged : {'Yes' if output.convergence_achieved else 'No'}")
    print(f"📊 Confidence: {output.confidence_score:.1f}%")
    print()
    print("✅ FINAL ANSWER:")
    print(f"   {output.final_answer}")
    print()

    if output.contradictions:
        print("⚠️  CONTRADICTIONS DETECTED:")
        for c in output.contradictions:
            print(f"   • {c}")
        print()

    if verbose:
        print("🔍 REASONING TRACE:")
        print(f"   {output.reasoning_trace}")
        print()
        print("🤖 AGENT RESPONSES:")
        for r in output.agent_responses:
            print(f"\n   [{r['agent_name']}]")
            print(f"   Answer    : {r['answer'][:200]}...")
            print(f"   Confidence: {r['confidence']}")

    print(f"\n⏱  Duration: {output.duration_seconds:.2f}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Conflict AI — Multi-agent adversarial reasoning"
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="The question to analyze")
    parser.add_argument("--context", "-c", type=str, default=None, help="Optional retrieval context")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full reasoning trace")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    config = load_config()
    manager = DebateManager(config)

    print(f"\n🔍 Processing query: '{args.query}'")
    print("   Running multi-agent debate...\n")

    output = manager.run(args.query, retrieval_context=args.context)
    record_output(output)

    if args.json:
        print(json.dumps(output.to_dict(), indent=2))
    else:
        print_output(output, verbose=args.verbose)


if __name__ == "__main__":
    main()
