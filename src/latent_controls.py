"""Controls share the exact generated H; no regeneration disguised as replay."""
from dataclasses import replace
import time
import numpy as np
import latent_backend as lb
import latent_handoff as lh
import latent_metrics as lm


def run_controls_v2(backend, packets, by_id, controls, band, dtype, pool, infra, store=None):
    import run_latent_reusability as run
    rows = []
    intervention_pairs = {}
    needs_h = any(c not in ("direct_packet", "closed_book") for c in controls)
    if needs_h:
        for p in packets:
            key = (p.packet_id, "latent_sb")
            if key not in pool:
                op = lambda p=p: run.encode(backend, "latent_sb", p, band, band.target)
                pool[key] = store.encoding({"control_H": p.record(), "source": p.packet_text,
                              "announced": p.announced_question, "band": run.as_band(band)}, op) if store else op()
    for packet in packets:
        if store:
            store.check_deadline()
        cards = run.cards_for(by_id[packet.dossier_id])
        if packet.evaluation_ids:
            cards = [c for c in cards if c.fact_id in packet.evaluation_ids]
        own = pool.get((packet.packet_id, "latent_sb"))
        for control in controls:
            if control in run.INFRASTRUCTURE_CONTROLS and packet.dossier_id not in infra:
                continue
            schema = ""
            vectors = None
            alignment = "none"
            error = ""
            text = ""
            cost = lm.CostBreakdown()
            diagnostics = {}
            if control == "direct_packet":
                text = packet.packet_text
            elif control == "closed_book":
                pass
            elif control == "transcript_text":
                text = own.transcript.text
            elif control in ("token_replay", "transcript_tail"):
                text = own.transcript.tail_text
            elif control == "embed_replay":
                # Exact same tokenization as token_replay, exact same schema.
                vectors = backend.embed_text(own.transcript.tail_text)
                alignment = "input-embedding-replay"
            elif control in ("source_free", "fixed_fact_swap", "fixed_fact_sham"):
                prompt = "Continue the fixed research notes below."
                if packet.announced_question:
                    prompt += "\nThe next agent's current question is: " + packet.announced_question
                if control != "source_free":
                    from latent_interventions import prepare_pair
                    if packet.packet_id not in intervention_pairs:
                        intervention_pairs[packet.packet_id] = prepare_pair(backend,packet,
                            run.cards_for(by_id[packet.dossier_id]),own.transcript)
                    pair = intervention_pairs[packet.packet_id]
                    if not pair["available"]:
                        error = "intervention_unavailable: " + pair["reason"]
                    else:
                        prompt = pair["swap_prompt" if control == "fixed_fact_swap" else "sham_prompt"]
                    diagnostics = {key:value for key,value in pair.items() if not key.endswith("_prompt")}
                replay = own.transcript
                if error:
                    vectors = np.zeros_like(own.vectors)
                else:
                    t0 = time.perf_counter()
                    replay = backend.teacher_force(prompt, own.transcript,
                                                    own.transcript.requested_k, system=run.WRITER_SYSTEM)
                    cost.sender_prefill_s = time.perf_counter()-t0
                    t0 = time.perf_counter()
                    vectors = run._align(backend, replay.states, replay.reference_embeddings)
                    cost.align_s = time.perf_counter()-t0
                schema, alignment = run.LATENT_SCHEMA, lb.ALIGNMENT_VERSION
                diagnostics.update(fixed_transcript=replay.text == own.transcript.text,
                                   same_token_ids=replay.token_ids == own.transcript.token_ids,
                                   alignment=dict(getattr(backend,"last_alignment",{})))
            elif control == "zero_prefix":
                vectors = np.zeros_like(own.vectors)
                schema, alignment = run.LATENT_SCHEMA, lb.ALIGNMENT_VERSION
            elif control == "shuffled_prefix":
                candidates = [p for p in packets if p.dossier_id != packet.dossier_id
                              and len(pool[(p.packet_id, "latent_sb")].vectors) >= len(own.vectors)]
                schema, alignment = run.LATENT_SCHEMA, lb.ALIGNMENT_VERSION
                if not candidates:
                    vectors = np.zeros_like(own.vectors)
                    error = "no_cross_dossier_same_length_donor"
                else:
                    donor = candidates[0]
                    vectors = pool[(donor.packet_id, "latent_sb")].vectors[:len(own.vectors)].copy()
                    diagnostics = {"donor_dossier_id": donor.dossier_id}
            else:
                raise ValueError(f"unknown control {control}")
            channel = "latent_sb" if vectors is not None else "text_prose"
            enc = run.Encoded(text, lm.PositionAccount(
                    payload=len(vectors) if vectors is not None else backend.count_positions(text),
                    schema=backend.count_positions(schema+"\n") if schema else 0),
                    cost, vectors=vectors, alignment=alignment, diagnostics=diagnostics,
                    error=error)
            for card in cards:
                payload = run.seal(channel, packet, card.question, enc, dtype, tag=control)
                payload = lh.from_wire(lh.to_wire(payload))
                current_cost = lm.CostBreakdown(**vars(cost))
                if error:
                    gen = lb.Generation("", lb.Usage(calls=0))
                elif store:
                    gen = store.read(backend, payload, lambda: run.answer(backend, payload, packet, lm.CostBreakdown()))
                    current_cost.reader_prefill_s += gen.usage.prefill_s
                    current_cost.reader_decode_s += gen.usage.decode_s
                else:
                    gen = run.answer(backend, payload, packet, current_cost)
                row = run._answer_record(packet, control, card.fact_id, card.question,
                          card.golds, gen, payload, enc, current_cost, band, control)
                row["control_diagnostics"] = diagnostics
                if diagnostics.get("available") and card.fact_id == diagnostics.get("target_fact_id"):
                    row["follows_changed_fact"] = run.score_row(gen.text,[diagnostics["changed_gold"]])["em"]
                rows.append(row)
                if store:
                    store.append("answers", row)
    return rows
