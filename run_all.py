"""Run all lab steps sequentially, or a single step with --step N."""

import argparse
import importlib
import sys
import time


STEPS = {
    1: ("01_langsmith_rag_pipeline", "Step 1: LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing",  "Step 2: Prompt Hub A/B Routing"),
    3: ("03_ragas_evaluation",       "Step 3: RAGAS Evaluation (~15-30 min)"),
    4: ("04_guardrails_validator",   "Step 4: Guardrails AI Validators"),
}


def run_step(n: int):
    module_name, label = STEPS[n]
    print(f"\n{'#' * 70}")
    print(f"#  {label}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n⏱  Step {n} finished in {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Run Day 22 lab steps.")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Run only this step (1-4). Omit to run all.")
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else [1, 2, 3, 4]

    for n in steps_to_run:
        run_step(n)

    print("🎉 All requested steps completed.")


if __name__ == "__main__":
    main()
