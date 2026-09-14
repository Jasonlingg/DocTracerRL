"""Generate teacher trajectories using Claude Sonnet 5 for SFT."""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import anthropic

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.agent import SYSTEM_PROMPT, run_question

class AnthropicPolicy:
    """An agent policy driven by Claude Sonnet 5 to generate SFT data."""
    
    def __init__(self, max_tokens: int = 4096):
        load_dotenv(override=True)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "sk-ant-...":
            raise ValueError("Please set a valid ANTHROPIC_API_KEY in the .env file")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-5"
        self.config = {
            "backend": "anthropic_api",
            "model": self.model,
            "max_tokens": max_tokens,
            "revision_note": "Teacher model for distillation data collection"
        }
        self.history = []

    def act(self, observation: str) -> str:
        self.history.append({
            "role": "user",
            "content": observation
        })
        
        sys_prompt = SYSTEM_PROMPT.strip() + "\n\nIMPORTANT: You must output ONLY RAW JSON. Do not write markdown, ```json, or any thought blocks. Just the raw JSON object."

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config["max_tokens"],
                system=sys_prompt,
                messages=self.history
            )
            
            # Look through content blocks for text. Sonnet 5 native APIs may include ThinkingBlocks.
            # We want the final text block.
            text_blocks = [blk.text for blk in response.content if hasattr(blk, 'text')]
            action = text_blocks[-1] if text_blocks else ""
            
            if action.startswith("```json"): action = action[7:]
            if action.startswith("```"): action = action[3:]
            if action.endswith("```"): action = action[:-3]
            action = action.strip()
            
            self.history.append({
                "role": "assistant",
                "content": action
            })
            
            return action
            
        except Exception as e:
            print(f"Anthropic API Error: {e}", file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description="Generate Golden Teacher Data using Claude 3.5")
    parser.add_argument("--snapshot", type=Path, default=Path("out/research/starter-2026-09-12"),
                        help="Path to the frozen document snapshot")
    parser.add_argument("--questions", type=Path, default=Path("data/research/questions.json"),
                        help="Path to questions list")
    parser.add_argument("--out-dir", type=Path, default=Path("out/research/teacher_trajectories"),
                        help="Output directory for generated trajectories")
    parser.add_argument("--max-steps", type=int, default=8, help="Max interaction steps")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(args.questions.read_text())["questions"]
    
    print(f"Starting teacher data generation for {len(questions)} questions...")
    
    success_count = 0
    for q in questions:
        q_id = q["id"]
        out_file = args.out_dir / f"teacher_{q_id}.json"
        
        if out_file.exists():
            out_file.unlink()
        md_file = out_file.with_suffix(".md")
        if md_file.exists():
            md_file.unlink()
            
        print(f"\n[{q_id}] Running Claude Sonnet 5 on: '{q['question']}'")
        policy = AnthropicPolicy()
        
        try:
            result = run_question(
                snapshot=args.snapshot,
                question=q,
                policy=policy,
                output=out_file,
                max_steps=args.max_steps,
                server_hardware="anthropic_cloud"
            )
            
            if result["status"] == "submitted":
                if result.get("checks"):
                    valid_sources = result["checks"]["claims_with_valid_source_spans"]
                    invalid_quotes = result["checks"]["invalid_quote_count"]
                    
                    if invalid_quotes == 0 and valid_sources == result["checks"]["claim_count"]:
                        print(f"  -> SUCCESS! Trajectory saved to {out_file.name}")
                        success_count += 1
                    else:
                        print(f"  -> WARNING: Submitted, but failed exact-quote verification eval.")
                else:
                    print(f"  -> Submitted, but missing checks array.")
            else:
                print(f"  -> FAILED: Hit error or max_steps. Check {out_file.name} for traceback.")
        except Exception as e:
            print(f"  -> Exception caught: {e}")
            
    print(f"\nDone! Generated {success_count} successful and verified golden trajectories.")

if __name__ == "__main__":
    raise SystemExit(main())