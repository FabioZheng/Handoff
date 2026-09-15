"""Full-source development panel with fixed current/hidden query assignments."""
from dataclasses import dataclass
import fictional_qa as fqa
import regret_data as rd
import latent_handoff as lh


@dataclass(frozen=True)
class RelationDossier:
    dossier_id: str
    cards: tuple
    document: str


def full_source_panel(path):
    from run_latent_reusability import Packet
    manifest, contexts = rd.load_contexts(path)
    packets, dossiers = [], []
    for context in contexts:
        rd.validate_relation_context(context)
        cards = tuple(fqa.EvidenceCard("", q.qid,
                      fqa._source_sentence(context.source, q.golds, q.qid),
                      q.question, q.golds, q.role) for q in context.questions)
        dossiers.append(RelationDossier(context.context_id, cards, context.source))
        anchors = [q for q in context.questions if q.role == rd.ANCHOR]
        for index, current in enumerate(anchors):
            paraphrase = next(q for q in context.questions if q.aspect == current.aspect and q.role == rd.PARAPHRASE)
            nearby = next(q for q in context.questions if q.aspect == current.aspect and q.role == rd.SAME_ENTITY)
            orthogonal = anchors[(index+1) % len(anchors)]
            chosen = (current, paraphrase, nearby, orthogonal)
            labels = tuple((q.qid,label) for q,label in zip(chosen, ("current","paraphrase","nearby","orthogonal")))
            for mode in ("generic", "conditioned"):
                packets.append(Packet(lh.opaque_id("B",context.context_id,current.qid,mode),
                    context.context_id, current.qid, mode,
                    current.question if mode == "conditioned" else "",
                    tuple(q.qid for q in context.questions), context.source,
                    lh.canonical_hash(context.source), (), (), panel="B",
                    evaluation_ids=tuple(q.qid for q in chosen),
                    current_fact_id=current.qid, relation_labels=labels))
    return dossiers, packets, manifest
