"""Gemini provider for routing-first harvest labels."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from data_harvest.llm.provider import RelabelCandidate, RelabelResult


class GeminiProvider:
    """Google Gemini provider using REST API."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-preview",
        temperature: float = 0.0,
        timeout_sec: int = 20,
        max_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GENAI_API_KEY) is not set.")

    def relabel(self, sample_payload: dict[str, Any]) -> RelabelResult:
        text_prompt = self._build_prompt(sample_payload)
        request_body = {
            "contents": [
                {
                    "parts": [
                        {"text": text_prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        pre_image_b64 = sample_payload.get("pre_image_b64")
        if pre_image_b64:
            request_body["contents"][0]["parts"].append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": pre_image_b64,
                    }
                }
            )

        resp = self._request_with_retry(request_body)
        text = self._extract_text(resp)
        parsed = self._parse_json(text)
        return self._to_result(parsed, raw_text=text)

    def _request_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err: Exception | None = None
        for i in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_err = exc
                if i < self.max_retries:
                    time.sleep(0.8 * (2 ** i))
                    continue
        raise RuntimeError(f"Gemini request failed: {last_err}")

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                return p["text"]
        raise RuntimeError("Gemini response has no text part.")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)

    @staticmethod
    def _to_result(parsed: dict[str, Any], raw_text: str) -> RelabelResult:
        chosen = parsed.get("chosen", {})
        chosen_candidate = RelabelCandidate(
            rank=1,
            bbox_xyxy=[float(x) for x in chosen["bbox_xyxy"]] if "bbox_xyxy" in chosen else None,
            semantic_text=chosen.get("semantic_text"),
            semantic_id=chosen.get("semantic_id"),
            function_id=chosen.get("function_id"),
            primitive_id=chosen.get("primitive_id"),
            screen_type=chosen.get("screen_type"),
            situation_id=chosen.get("situation_id"),
            roi_name=chosen.get("roi_name"),
            roi_bbox_norm=[float(x) for x in chosen["roi_bbox_norm"]] if "roi_bbox_norm" in chosen else None,
            action=chosen.get("action"),
            confidence=float(chosen.get("confidence", 0.0)),
            source="gemini",
        )

        out_candidates: list[RelabelCandidate] = []
        for idx, c in enumerate(parsed.get("candidates", []), start=1):
            try:
                out_candidates.append(
                    RelabelCandidate(
                        rank=int(c.get("rank", idx)),
                        bbox_xyxy=[float(x) for x in c["bbox_xyxy"]] if "bbox_xyxy" in c else None,
                        semantic_text=c.get("semantic_text"),
                        semantic_id=c.get("semantic_id"),
                        function_id=c.get("function_id"),
                        primitive_id=c.get("primitive_id"),
                        screen_type=c.get("screen_type"),
                        situation_id=c.get("situation_id"),
                        roi_name=c.get("roi_name"),
                        roi_bbox_norm=[float(x) for x in c["roi_bbox_norm"]] if "roi_bbox_norm" in c else None,
                        action=c.get("action"),
                        confidence=float(c.get("confidence", 0.0)),
                        source=str(c.get("source", "gemini")),
                    )
                )
            except Exception:
                continue

        out_candidates.sort(
            key=lambda candidate: (
                candidate.rank if candidate.rank is not None else 999,
                -float(candidate.confidence or 0.0),
            )
        )
        for idx, candidate in enumerate(out_candidates, start=1):
            if candidate.rank is None or candidate.rank < 1:
                candidate.rank = idx

        evidence = parsed.get("evidence", {})
        evidence["raw_response_text"] = raw_text
        return RelabelResult(chosen=chosen_candidate, candidates=out_candidates, evidence=evidence)

    @staticmethod
    def _build_prompt(sample_payload: dict[str, Any]) -> str:
        taxonomy = sample_payload.get("routing_taxonomy", {})
        prompt_payload = {
            key: value
            for key, value in sample_payload.items()
            if key not in {"pre_image_b64", "ontology", "routing_taxonomy"}
        }
        return (
            "You are labeling a Civilization VI page for primitive routing.\n"
            "Return strict JSON only with keys: chosen, candidates, evidence.\n"
            "chosen={screen_type, situation_id, primitive_id, roi_name, confidence}\n"
            "candidates=[{rank, screen_type, situation_id, primitive_id, roi_name, confidence, source}]\n"
            "evidence={matched_must_have,matched_strong_cues,triggered_hard_negatives,conflict_pair,open_screen_detected,reasoning}\n"
            "Task: choose exactly one primitive_id for the current page from the provided taxonomy.\n"
            "Return up to 3 ranked candidates in candidates. rank=1 is best, rank=2 is second, rank=3 is third.\n"
            "The chosen object must match rank 1.\n"
            "Use only visible UI evidence from the screenshot. Do not use strategy, intent, or hidden game state.\n"
            "Prioritize layout, panel structure, button placement, repeated card/list patterns, and obvious screen-specific visuals.\n"
            "Apply this decision procedure strictly:\n"
            "1. If a dedicated full screen or dedicated panel is actually open, do NOT label as popup_primitive.\n"
            "2. Use each primitive's hard_negatives to remove candidates before final choice.\n"
            "3. A primitive with unmet must_have_visuals should be ranked below one with satisfied must_have_visuals.\n"
            "4. Distinguish entry buttons from opened screens:\n"
            "   - only entry button visible -> popup_primitive\n"
            "   - actual research/city production/civic/policy/religion/governor/etc screen open -> dedicated primitive\n"
            "5. unit_ops_primitive means non-combat general unit control only.\n"
            "6. combat_primitive means immediate fight decision where enemy target + HP/combat state are visually central.\n"
            "7. policy change popup belongs to policy_primitive, not popup_primitive.\n"
            "8. Always return one best guess. No ambiguous class.\n"
            "Choose roi_name from the provided ROI taxonomy when possible. This is for router_roi cropping.\n"
            "Do not return bbox, semantic_id, function_id, or other grounding fields.\n"
            f"Sample:\n{json.dumps(prompt_payload, ensure_ascii=False)}\n"
            f"Routing taxonomy:\n{json.dumps(taxonomy, ensure_ascii=False)}\n"
        )

    @staticmethod
    def image_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    @staticmethod
    def _load_dotenv() -> None:
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
