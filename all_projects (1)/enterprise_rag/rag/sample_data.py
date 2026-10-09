"""Synthetic corpus + eval set so the pipeline can be exercised end to end without private data.

Each document is a short project brief with a unique code name, owner, budget and launch quarter,
padded with generic filler. Questions ask for one fact about one project, so the relevant
document is known exactly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ADJ = "amber azure brisk cobalt crimson dusky ember frosty gilded hazel indigo jade lunar misty nimble opal quartz russet silver tawny".split()
NOUN = "falcon harbor meadow comet ridge lantern orchard summit voyager willow anchor beacon canyon delta engine forge glacier horizon island junction".split()
FIRST = "Aarav Priya Maria Chen Omar Lena Diego Hana Ivan Sofia Kofi Mei Noah Zara Luca".split()
LAST = "Sharma Garcia Wang Haddad Novak Okafor Silva Tanaka Rossi Petrov Nguyen Khan Muller Lopez Reddy".split()
DEPT = ["Finance", "Operations", "Legal", "Engineering", "Procurement", "Customer Success", "Security"]
QUARTERS = [f"Q{q} {y}" for y in (2026, 2027) for q in (1, 2, 3, 4)]
FILLER = [
    "All milestones are reviewed by the steering committee on a monthly cadence.",
    "Vendor onboarding must follow the standard procurement checklist before any contract is signed.",
    "Data handled by the initiative is classified as internal and must not leave approved systems.",
    "Status reports are due on the first business day of each month.",
    "Any deviation above five percent of plan requires written approval from the department head.",
    "The team maintains a risk register that is reviewed during every sprint planning session.",
    "Training sessions for affected staff are scheduled two weeks before go-live.",
    "Access to the shared workspace is granted through the identity management portal.",
    "Post-launch support is covered by the regional service desk for the first ninety days.",
    "Quarterly audits verify that documentation matches the deployed configuration.",
    "Change requests are logged, triaged, and prioritised weekly.",
    "Dependencies on other programmes are tracked in the portfolio dashboard.",
]


def generate(out_dir: str | Path, n_docs: int = 200, seed: int = 0) -> Path:
    """Write `n_docs` .md files to out_dir and eval.jsonl next to it. Returns the eval file path."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_rows = []
    for i in range(n_docs):
        code = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}-{i:04d}"
        owner = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        budget = rng.randint(10, 900) * 1000
        launch = rng.choice(QUARTERS)
        dept = rng.choice(DEPT)
        paras = [
            f"# Project {code}\n\nProject {code} is an initiative of the {dept} department. "
            f"The project owner is {owner}. The approved budget is ${budget:,}. "
            f"Launch is planned for {launch}.",
            " ".join(rng.sample(FILLER, 4)),
            " ".join(rng.sample(FILLER, 4)),
        ]
        name = f"project_{i:04d}.md"
        (out / name).write_text("\n\n".join(paras), encoding="utf-8")
        eval_rows += [
            {"question": f"Who is the owner of project {code}?", "relevant_doc_ids": [name]},
            {"question": f"What is the approved budget for project {code}?", "relevant_doc_ids": [name]},
            {"question": f"When is the launch of project {code} planned?", "relevant_doc_ids": [name]},
        ]
    eval_path = out.parent / "eval.jsonl" if out.name else out / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_rows), encoding="utf-8")
    return eval_path
