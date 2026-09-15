"""Fixed-transcript, fixed-position factual substitution versus irrelevant sham.

Eligibility is logged, never used to select primary evaluation sources. Only
facts in the encoder's actual input can be intervened on. Hidden evaluator
golds help choose an intervention but are never given to the reader.
"""
from dataclasses import replace
import latent_handoff as lh


def matched_alternative(backend, source, value):
    if source.count(value) != 1 or len(value) < 2:
        return None
    candidates = [value[i:]+value[:i] for i in range(1,min(len(value),20))]
    candidates += ["".join(str((int(c)+n)%10) if c.isdigit() else c for c in value) for n in range(1,10)]
    size = backend.count_positions(source)
    for candidate in candidates:
        if candidate.casefold() == value.casefold() or candidate in source:
            continue
        changed = source.replace(value,candidate)
        if len(changed) == len(source) and backend.count_positions(changed) == size:
            return candidate
    return None


def prepare_pair(backend, packet, cards, transcript):
    """Returns matched prompts, target question and target new value, or reason."""
    from run_latent_reusability import writer_prompt
    import latent_metrics as lm
    band = lm.PositionBand(backend.config.transcript_max_tokens,0.85)
    original_prompt = writer_prompt("latent_sb",packet,band)
    candidates = []
    for card in cards:
        if card.fact_id not in packet.selected_fact_ids or card.question == packet.announced_question:
            continue
        if any(g.casefold() in transcript.text.casefold() for g in card.golds):
            continue
        for gold in card.golds:
            alt = matched_alternative(backend,packet.packet_text,gold)
            if alt is not None:
                prompt = writer_prompt("latent_sb",replace(packet,packet_text=packet.packet_text.replace(gold,alt)),band)
                if backend.count_positions(prompt) == backend.count_positions(original_prompt):
                    candidates.append((card,gold,alt,prompt))
                    break
    if len(candidates) < 2:
        return {"available":False,"reason":"need two source facts absent from H with length-matched alternatives"}
    target, sham = candidates[:2]
    return {"available":True,"target_fact_id":target[0].fact_id,
            "original_golds":list(target[0].golds),"changed_gold":target[2],
            "swap_prompt":target[3],"sham_prompt":sham[3],
            "original_prompt_positions":backend.count_positions(original_prompt),
            "swap_prompt_positions":backend.count_positions(target[3]),
            "sham_prompt_positions":backend.count_positions(sham[3]),
            "transcript_hash":lh.canonical_hash([transcript.text,transcript.token_ids])}
