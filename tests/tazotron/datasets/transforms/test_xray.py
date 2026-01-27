import pytest
import torch
import torchio as tio

from tazotron.datasets.transforms.xray import RenderDRR, RenderDRRConfig


def _make_image(fill: float = 1.0) -> tio.ScalarImage:
    tensor = torch.full((1, 2, 2, 2), fill_value=fill, dtype=torch.float32)
    return tio.ScalarImage(tensor=tensor, affine=torch.eye(4))


def _make_subject(
    *,
    include_reorient: bool = True,
    rotations: torch.Tensor | None = None,
    translations: torch.Tensor | None = None,
) -> tio.Subject:
    volume = _make_image(1.0)
    density = _make_image(0.5)
    subject_dict: dict[str, object] = {"volume": volume, "density": density}
    if include_reorient:
        subject_dict["reorient"] = _make_image(1.0)
    subject = tio.Subject(subject_dict)
    if rotations is not None:
        subject["rotations"] = rotations
    if translations is not None:
        subject["translations"] = translations
    return subject


class _DummyDRR:
    last_instance: "_DummyDRR | None" = None
    output: torch.Tensor = torch.zeros(1, 1, 2, 2, dtype=torch.float32)

    def __init__(self, subject: tio.Subject, **kwargs: object) -> None:
        self.subject = subject
        self.kwargs = kwargs
        self.to_device: torch.device | None = None
        self.call_args: tuple[torch.Tensor, torch.Tensor, dict[str, object]] | None = None
        _DummyDRR.last_instance = self

    def to(self, device: torch.device | str) -> "_DummyDRR":
        self.to_device = torch.device(device)
        return self

    def __call__(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        self.call_args = (rotations, translations, kwargs)
        return self.output


class TestRenderDRR:
    @pytest.mark.fast
    def test_config_resolved_device_handles_string(self) -> None:
        config = RenderDRRConfig(device="cpu")
        resolved = config.resolved_device()
        assert isinstance(resolved, torch.device)
        assert resolved.type == "cpu"

    @pytest.mark.fast
    def test_raises_for_non_subject_input(self) -> None:
        transform = RenderDRR()
        with pytest.raises(TypeError, match="torchio.Subject"):
            transform({"not": "a subject"})  # type: ignore[arg-type]

    @pytest.mark.fast
    def test_raises_when_required_diffdrr_keys_missing(self) -> None:
        subject = _make_subject(include_reorient=False)
        transform = RenderDRR()
        with pytest.raises(KeyError, match="diffdrr.read"):
            transform(subject)

    @pytest.mark.fast
    def test_raises_when_pose_is_missing(self) -> None:
        subject = _make_subject(include_reorient=True)
        transform = RenderDRR()
        with pytest.raises(ValueError, match="Both rotations and translations"):
            transform(subject)

    @pytest.mark.fast
    def test_raises_for_invalid_rotation_shape(self) -> None:
        subject = _make_subject(
            rotations=torch.zeros(3, dtype=torch.float32),
            translations=torch.zeros((1, 3), dtype=torch.float32),
        )
        transform = RenderDRR()
        with pytest.raises(ValueError, match=r"rotations must have shape \(B, 3\)"):
            transform(subject)

    @pytest.mark.fast
    def test_raises_for_mismatched_pose_batch_sizes(self) -> None:
        subject = _make_subject(
            rotations=torch.zeros((2, 3), dtype=torch.float32),
            translations=torch.zeros((1, 3), dtype=torch.float32),
        )
        transform = RenderDRR()
        with pytest.raises(ValueError, match="matching batch sizes"):
            transform(subject)

    @pytest.mark.fast
    def test_raises_when_rendered_drr_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)
        _DummyDRR.output = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
        subject = _make_subject(
            rotations=torch.zeros((1, 3), dtype=torch.float32),
            translations=torch.zeros((1, 3), dtype=torch.float32),
        )
        transform = RenderDRR()
        with pytest.raises(ValueError, match="Rendered DRR is empty"):
            transform(subject)

    @pytest.mark.fast
    def test_sets_xray_on_successful_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)
        _DummyDRR.output = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]]]],
            dtype=torch.float32,
        )
        subject = _make_subject(
            rotations=torch.zeros((1, 3), dtype=torch.float32),
            translations=torch.zeros((1, 3), dtype=torch.float32),
        )
        transform = RenderDRR({"device": "cpu", "height": 2, "delx": 1.0})
        rendered = transform(subject)

        assert "xray" in rendered
        assert torch.equal(rendered["xray"], _DummyDRR.output)

        dummy = _DummyDRR.last_instance
        assert dummy is not None
        assert dummy.to_device == torch.device("cpu")
        rotations_arg, translations_arg, kwargs = dummy.call_args or (None, None, {})
        assert rotations_arg is not None and translations_arg is not None
        assert rotations_arg.device.type == "cpu"
        assert translations_arg.device.type == "cpu"
        assert kwargs.get("parameterization") == "euler_angles"
        assert kwargs.get("convention") == "ZXY"
