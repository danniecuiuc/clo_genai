from __future__ import annotations

from textwrap import dedent


def build_memo_prompt(tranche_payload: dict, pricing_result: dict, comps_summary: list[dict]) -> str:
    return dedent(
        f'''
        You are generating a short CLO tranche pricing memo.

        Tranche payload:
        {tranche_payload}

        Pricing result:
        {pricing_result}

        Comparable deals:
        {comps_summary}

        Write a concise memo with:
        1. estimated spread / pricing range
        2. key drivers
        3. comp set summary
        4. uncertainty / caveats
        '''
    ).strip()


def sample_local_memo(tranche_payload: dict, pricing_result: dict, comps_summary: list[dict]) -> str:
    """Simple offline fallback so the demo still shows a memo section."""
    comp_ids = [row.get('Bloomberg ID', 'unknown') for row in comps_summary[:3]]
    return (
        f"Estimated spread is {pricing_result.get('prediction', 'n/a'):.2f} with range "
        f"[{pricing_result.get('lower_bound', 'n/a'):.2f}, {pricing_result.get('upper_bound', 'n/a'):.2f}]. "
        f"Top comparable deals include {', '.join(comp_ids)}. "
        f"Uncertainty flag is {pricing_result.get('uncertainty_flag', 'n/a')}."
    )
