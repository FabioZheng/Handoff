"""Real PyTorch/Transformers path, tiny random weights, no network or GPU.

This checks implementation correctness, not task accuracy or model quality.
"""
import json
from types import SimpleNamespace
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import latent_backend as lb
from latent_alignment import AlignmentConfig, statebridge_align


class TinyTokenizer:
    all_special_ids = [0, 1]
    eos_token_id = 0
    pad_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 256 + 2 for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long) if return_tensors else ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)-2) for i in ids if int(i) >= 2)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)+"<assistant>\n"


def main():
    torch.set_num_threads(2)
    torch.manual_seed(22)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=258, hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=4096,
                eos_token_id=0, pad_token_id=1))
    backend = lb.TransformersBackend(lb.BackendConfig(model_id="test/random", revision="test",
                    device="cpu", dtype="float32", answer_max_tokens=4, transcript_max_tokens=8),
                    model=model, tokenizer=TinyTokenizer())
    results = []
    def check(label, ok):
        assert ok, label
        results.append(label)
        print("PASS", label)
    parity = backend.wrapper_parity("Fictional value: qrz.", "What is the value?", atol=1e-6)
    check("production reader layout and cached generation parity", parity["passed"])
    check("schema-bearing production layout parity", backend.wrapper_parity("field: qrz", "Value?", 1e-6, "field: value")["passed"])
    h = backend.transcript_states("EVIDENCE: planet has code qrz.", 4)
    check("states pair with valid token references", h.states.shape == h.reference_embeddings.shape and len(h.states) == min(4, len(h.token_ids)))
    replay = backend.teacher_force("EVIDENCE: planet has code qrz.", h, 4)
    check("teacher forcing reproduces greedy predicting states", np.allclose(h.states, replay.states, atol=1e-5))
    check("teacher forcing preserves exact H token IDs", h.token_ids == replay.token_ids and h.text == replay.text)
    replay2 = backend.teacher_force("EVIDENCE: planet has code abc.", h, 4)
    check("fixed transcript still responds to source intervention", not np.allclose(h.states, replay2.states, atol=1e-7))
    aligned = backend.align(h.states, h.reference_embeddings)
    check("full alignment yields finite continuous prefix", aligned.shape == h.states.shape and np.isfinite(aligned).all())
    check("aligned prefix is consumed by production decoder", backend.read_message(aligned, "Value?").usage.generated_tokens > 0)
    check("receiver state is fresh after unrelated generation", backend.read_message("qrz", "Value?").text == backend.read_message("qrz", "Value?").text)

    # Geometry invariant: an orthogonal coordinate change with paired references
    # must align back to those references when covariances are well conditioned.
    torch.manual_seed(3)
    ref = torch.randn(80, 8)
    q, _ = torch.linalg.qr(torch.randn(8,8))
    q[:,0] *= torch.linalg.det(q)
    states = ref @ q + 7
    vocab = torch.randn(43,8)
    a, _ = statebridge_align(states, ref, vocab, AlignmentConfig(snap_ratio=0))
    expected = ref * (vocab.norm(dim=1).mean() / ref.norm(dim=1,keepdim=True))
    check("known proper rotation and offset are removed", torch.allclose(a, expected, atol=3e-4))
    b, _ = statebridge_align(states, ref, vocab, AlignmentConfig(vocab_chunk=7))
    c, _ = statebridge_align(states, ref, vocab, AlignmentConfig(vocab_chunk=1000))
    check("chunked vocabulary anchoring matches full search", torch.allclose(b,c,atol=1e-5))
    d, info = statebridge_align(torch.zeros(4,8), torch.ones(4,8), vocab)
    check("rank-deficient repeated states remain finite", torch.isfinite(d).all().item())
    check("prefill and decode are separately measured", h.usage.prefill_s > 0 and h.usage.decode_s > 0)
    print(json.dumps({"checks":len(results), "backend":"tiny-random-cpu", "scientific_result":False}))


if __name__ == "__main__":
    main()
