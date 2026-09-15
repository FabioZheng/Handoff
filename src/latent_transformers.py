"""Fresh-state, greedy local inference for the Experiment 13 protocol.

No model download occurs on import. A tiny random model/tokenizer can be
injected for integration tests of the actual inference path on CPU.
"""
import time
import numpy as np
from latent_backend import Backend, BackendConfig, Generation, Transcript, Usage, align_states
from latent_alignment import AlignmentConfig, VERSION, NORM_VERSION, statebridge_align


class TransformersBackend(Backend):
    def __init__(self, config: BackendConfig, *, model=None, tokenizer=None):
        import torch
        self.config, self._torch = config, torch
        self._dtype = getattr(torch, config.dtype)
        started = time.perf_counter()
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not config.revision:
                raise ValueError("pin an exact checkpoint revision before loading weights")
            tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, revision=config.revision, dtype=self._dtype)
        if tokenizer is None:
            raise ValueError("a tokenizer is required")
        self.model, self.tokenizer = model.to(config.device).eval(), tokenizer
        self.model.requires_grad_(False)
        self._embeddings = self.model.get_input_embeddings()
        self._last_block = self.model.model.layers[-1]
        # Stream table statistics; no second full float32 vocabulary on GPU.
        sq, total, anchor = 0.0, 0, torch.zeros(self.hidden_width, device=config.device)
        with torch.inference_mode():
            for chunk in self._embeddings.weight.split(config.alignment_vocab_chunk):
                z = chunk.float()
                sq += float(z.square().sum())
                total += z.numel()
                anchor += z.sum(0)
        self._embed_rms = (sq / total) ** 0.5
        self._embed_anchor = (anchor / len(self._embeddings.weight)).cpu().numpy()
        self.load_seconds = time.perf_counter() - started
        self.last_alignment = {}

    @property
    def hidden_width(self):
        return int(self.model.config.hidden_size)

    def _clock(self):
        if self.config.device.startswith("cuda"):
            self._torch.cuda.synchronize()
        return time.perf_counter()

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.config.device)

    def _chat(self, prompt, system=None):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.config.enable_thinking)

    def count_positions(self, text):
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def embed_text(self, text):
        with self._torch.inference_mode():
            return self._embeddings(self._ids(text))[0].float().cpu().numpy().copy()

    def _decode(self, *, ids=None, embeds=None, cap=48, capture=False):
        """One prefill then cached greedy steps. Cache is local to this call.

        Capture each *predicting* state before the final model norm, paired
        with the token selected from that step's logits. EOS/special tokens
        are charged as generated but never transmitted as message states.
        """
        torch = self._torch
        if (ids is None) == (embeds is None) or cap < 1:
            raise ValueError("exactly one input representation and positive cap required")
        n = int((ids if ids is not None else embeds).shape[1])
        mask = torch.ones((1, n), device=self.config.device, dtype=torch.long)
        cache, tokens, rows, pending = None, [], [], []
        def hook(_module, _args, out):
            z = out[0] if isinstance(out, tuple) else out
            pending[:] = [z[0, -1].detach().clone()]
        handle = self._last_block.register_forward_hook(hook) if capture else None
        special = set(self.tokenizer.all_special_ids)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        first_s = decode_s = 0.0
        start = self._clock()
        if self.config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                for step in range(cap):
                    if time.monotonic() >= getattr(self, "deadline", float("inf")):
                        raise TimeoutError("experiment time envelope reached during decoding")
                    tick = self._clock()
                    # Explicit sequential positions for both input paths.
                    positions = torch.arange(n, device=self.config.device) if step == 0 else torch.tensor([n + step - 1], device=self.config.device)
                    kwargs = dict(attention_mask=mask, position_ids=positions.unsqueeze(0),
                                  past_key_values=cache, use_cache=True)
                    if step == 0:
                        kwargs.update(input_ids=ids) if ids is not None else kwargs.update(inputs_embeds=embeds)
                    else:
                        kwargs["input_ids"] = torch.tensor([[tokens[-1]]], device=self.config.device)
                    out = self.model(**kwargs)
                    token = int(out.logits[0, -1].argmax())
                    cache = out.past_key_values
                    tokens.append(token)
                    if capture and token not in special:
                        rows.append(pending[0])
                    elapsed = self._clock() - tick
                    if step == 0:
                        first_s += elapsed
                    else:
                        decode_s += elapsed
                    if token in eos:
                        break
                    mask = torch.cat([mask, torch.ones((1, 1), device=mask.device, dtype=mask.dtype)], 1)
                matrix = torch.stack(rows).float().cpu().numpy() if rows else np.empty((0, self.hidden_width), np.float32)
        finally:
            if handle:
                handle.remove()
        peak = torch.cuda.max_memory_allocated() if self.config.device.startswith("cuda") else 0
        usage = Usage(prefill_positions=n, generated_tokens=len(tokens),
                      seconds=self._clock() - start, prefill_s=first_s,
                      decode_s=decode_s, peak_memory_bytes=peak)
        return tokens, matrix, usage

    def generate(self, prompt, max_new_tokens=None, system=None):
        ids = self._ids(self._chat(prompt, system))
        tokens, _, usage = self._decode(ids=ids, cap=max_new_tokens or self.config.answer_max_tokens)
        return Generation(self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), usage)

    def _transcript(self, tokens, states, k, usage, cap):
        special = set(self.tokenizer.all_special_ids)
        clean = tuple(t for t in tokens if t not in special)
        if len(clean) != len(states):
            raise ValueError("token/state alignment drift")
        take = min(max(0, k), len(clean))
        tail_ids = clean[-take:] if take else ()
        with self._torch.inference_mode():
            ref = self._embeddings(self._torch.tensor(tail_ids, dtype=self._torch.long,
                              device=self.config.device)).float().cpu().numpy()
        return Transcript(text=self.tokenizer.decode(clean, skip_special_tokens=True),
                states=states[-take:].copy() if take else states[:0].copy(),
                realized_length=len(tokens), requested_k=k, usage=usage,
                truncated=len(tokens) >= cap, token_ids=clean,
                reference_embeddings=ref.copy(),
                tail_text=self.tokenizer.decode(tail_ids, skip_special_tokens=True))

    def transcript_states(self, prompt, k, max_new_tokens=None, system=None):
        cap = max_new_tokens or self.config.transcript_max_tokens
        tokens, states, usage = self._decode(ids=self._ids(self._chat(prompt, system)), cap=cap, capture=True)
        return self._transcript(tokens, states, k, usage, cap)

    def teacher_force(self, prompt, transcript, k, system=None):
        """Replay exactly H, with no generation and no access to old KV state."""
        torch = self._torch
        ids = self._ids(self._chat(prompt, system))
        message_ids = tuple(transcript.token_ids)
        if not message_ids and transcript.text:
            raise ValueError("exact replay requires the original transcript token IDs")
        captured = []
        start_index = ids.shape[1] - 1
        def hook(_module, _args, out):
            z = out[0] if isinstance(out, tuple) else out
            captured.append(z[0, start_index:start_index + len(message_ids)].detach().float().cpu().numpy().copy())
        handle = self._last_block.register_forward_hook(hook)
        start = self._clock()
        try:
            with torch.inference_mode():
                tail = torch.tensor([message_ids[:-1]], device=ids.device, dtype=torch.long)
                full = torch.cat([ids, tail], 1)
                self.model(input_ids=full, attention_mask=torch.ones_like(full),
                           position_ids=torch.arange(full.shape[1], device=ids.device).unsqueeze(0), use_cache=False)
        finally:
            handle.remove()
        usage = Usage(prefill_positions=int(full.shape[1]), generated_tokens=0,
                      seconds=self._clock() - start, state_extract_s=self._clock() - start)
        result = self._transcript(message_ids, captured[0], k, usage, max(1, len(message_ids)+1))
        result.text = transcript.text
        result.truncated = transcript.truncated
        return result

    def align(self, states, references=None, mode=None):
        torch = self._torch
        mode = mode or self.config.alignment
        if mode == NORM_VERSION:
            self.last_alignment = {"version": mode}
            return align_states(states, self._embed_rms, self._embed_anchor)
        if references is None:
            raise ValueError("StateBridge needs the paired decoded-token embeddings")
        cfg = AlignmentConfig(self.config.alignment_regularization,
                              self.config.alignment_eigen_floor,
                              self.config.alignment_snap_ratio,
                              self.config.alignment_vocab_chunk)
        with torch.inference_mode():
            z, self.last_alignment = statebridge_align(
                torch.as_tensor(states, device=self.config.device),
                torch.as_tensor(references, device=self.config.device),
                self._embeddings.weight, cfg)
        return z.cpu().numpy().copy()

    def _reader_segments(self, question):
        from latent_handoff import READER_INSTRUCTION
        marker = "<HANDOFF_INSERT_7619>"
        prompt = f"MESSAGE FROM THE OTHER AGENT:\n{marker}\n\nQUESTION: {question}\nANSWER:"
        rendered = self._chat(prompt, READER_INSTRUCTION)
        if rendered.count(marker) != 1:
            raise ValueError("reader marker must occur exactly once")
        head, tail = rendered.split(marker)
        return self._ids(head), self._ids(tail)

    def _reader_input(self, message, question, schema=""):
        """Identical segmentation for text, embeddings and replay controls.

        The message lives inside the user turn, before the question and chat
        assistant delimiter. Schema is independently tokenized and charged.
        """
        torch = self._torch
        head, tail = self._reader_segments(question)
        legend = self._ids(schema + "\n") if schema else head[:, :0]
        with torch.inference_mode():
            if isinstance(message, str):
                middle = self._ids(message)
                return torch.cat([head, legend, middle, tail], 1), None
            rows = torch.as_tensor(np.asarray(message), device=head.device,
                                   dtype=self._embeddings.weight.dtype).unsqueeze(0)
            return None, torch.cat([self._embeddings(head), self._embeddings(legend), rows,
                                    self._embeddings(tail)], 1)

    def read_message(self, message, question, schema=""):
        ids, embeds = self._reader_input(message, question, schema)
        tokens, _, usage = self._decode(ids=ids, embeds=embeds, cap=self.config.answer_max_tokens)
        if not isinstance(message, str):
            usage.prefix_positions = len(message)
        return Generation(self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), usage)

    def wrapper_parity(self, text, question, atol=2e-2, schema=""):
        """Exercise the production reader layout, logits AND cached decoding."""
        torch = self._torch
        ids, _ = self._reader_input(text, question, schema)
        _, embeds = self._reader_input(self.embed_text(text), question, schema)
        mask = torch.ones(ids.shape, dtype=torch.long, device=ids.device)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        with torch.inference_mode():
            a = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[0,-1].float()
            b = self.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False).logits[0,-1].float()
        gap = float((a-b).abs().max())
        x, _, _ = self._decode(ids=ids, cap=4)
        y, _, _ = self._decode(embeds=embeds, cap=4)
        return {"max_logit_gap": gap, "atol": atol,
                "same_argmax": int(a.argmax()) == int(b.argmax()),
                "same_generated_ids": x == y,
                "passed": gap <= atol and int(a.argmax()) == int(b.argmax()) and x == y}

    def generate_from_prefix(self, prefix, suffix, max_new_tokens=None, system=None):
        # Compatibility for callers outside the runner. All experiment arms
        # use read_message with an explicit question and public schema.
        from latent_backend import _question_block
        return self.read_message(prefix, _question_block(suffix))
