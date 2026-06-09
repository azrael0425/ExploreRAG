from app.schemas.evaluation import EvalAblationCreate, EvalRunCreate


def test_date_shaped_case_order_seed_is_valid() -> None:
    assert EvalRunCreate(case_order_seed=20260810).case_order_seed == 20260810
    assert EvalAblationCreate().case_order_seed == 20260810


def test_ablation_variant_order_must_be_complete() -> None:
    try:
        EvalAblationCreate(variant_order=["A", "B", "C", "C"])
    except ValueError as exc:
        assert "variant_order" in str(exc)
    else:
        raise AssertionError("duplicate variants must be rejected")
