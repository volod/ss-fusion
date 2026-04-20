from selfsuvis.pipeline.workflows.local import runner


def test_is_simple_agentic_audit_returns_true_for_low_branching_video():
    video_context = {
        "caption_segments": 2,
        "qwen_captions": [{"t_sec": float(i)} for i in range(20)],
        "ocr": [{"ocr_text": "road sign"} for _ in range(4)],
        "map": {"points": 10},
        "world_model_clips": 7,
        "unidrive_analysis": [],
        "multi_model_comparison": {},
    }

    assert runner._is_simple_agentic_audit(video_context) is True


def test_is_simple_agentic_audit_returns_false_for_complex_video():
    video_context = {
        "caption_segments": 5,
        "qwen_captions": [{"t_sec": float(i)} for i in range(24)],
        "ocr": [{"ocr_text": "lane"} for _ in range(12)],
        "map": {"points": 48},
        "world_model_clips": 12,
        "unidrive_analysis": [{"summary": "present"}],
        "multi_model_comparison": {"mean_qwen_unidrive_agreement": 0.81},
    }

    assert runner._is_simple_agentic_audit(video_context) is False


def test_is_valid_agentic_flow_analysis_accepts_simple_structured_output():
    text = """
## Flow Summary
Short summary.

## Highest-Risk Steps
- R Qwen detailed captioning can amplify stale context.

## Failure Propagation
- Wrong OCR or detection cues can bias later reasoning and synthesis.

## Mitigations
- Gate downstream prompts with confidence and contradiction checks.
""".strip()

    assert runner._is_valid_agentic_flow_analysis(text, simple=True) is True


def test_is_valid_agentic_flow_analysis_rejects_incomplete_simple_output():
    text = """
## Flow Summary
Short summary.

## Mitigations
- Add checks.
""".strip()

    assert runner._is_valid_agentic_flow_analysis(text, simple=True) is False
