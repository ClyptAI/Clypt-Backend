ARG BASE_IMAGE=lmsysorg/sglang:v0.5.10-rocm720-mi30x
FROM ${BASE_IMAGE}

ENV SGLANG_USE_AITER=0
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=1
ENV SGLANG_ENABLE_QUARK_QUANTIZATION=0
ENV USE_ROCM_AITER_ROPE_BACKEND=0

RUN python - <<'PY'
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/__init__.py")
text = path.read_text(encoding="utf-8")

if "Clypt ROCm patch: disable Quark quantization imports by default" not in text:
    text = text.replace("import builtins\nimport inspect\n", "import builtins\nimport inspect\nimport os\n")
    text = text.replace(
        "from sglang.srt.layers.quantization.quark.quark import QuarkConfig\n"
        "from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config\n",
        "# Clypt ROCm patch: disable Quark quantization imports by default.\n"
        "# The upstream v0.5.10 ROCm 7.2 MI300X image imports Quark at module import\n"
        "# time, which pulls an incompatible bundled AITER path and crashes before\n"
        "# non-Quark models can start. Qwen/Qwen3.6-35B-A3B is not a Quark checkpoint,\n"
        "# so keep Quark out of the registry unless explicitly enabled for validation.\n"
        "QuarkConfig = None\n"
        "QuarkInt4Fp8Config = None\n"
        "if os.getenv(\"SGLANG_ENABLE_QUARK_QUANTIZATION\", \"0\").lower() in {\"1\", \"true\", \"yes\", \"on\"}:\n"
        "    from sglang.srt.layers.quantization.quark.quark import QuarkConfig\n"
        "    from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config\n",
    )
    text = text.replace('    "quark": QuarkConfig,\n', "")
    text = text.replace('    "quark_int4fp8_moe": QuarkInt4Fp8Config,\n', "")
    text = text.replace(
        "\n\nif is_cuda() or (_is_mxfp_supported and is_hip()):\n",
        "\n\nif QuarkConfig is not None:\n"
        "    BASE_QUANTIZATION_METHODS[\"quark\"] = QuarkConfig\n"
        "if QuarkInt4Fp8Config is not None:\n"
        "    BASE_QUANTIZATION_METHODS[\"quark_int4fp8_moe\"] = QuarkInt4Fp8Config\n"
        "\n"
        "if is_cuda() or (_is_mxfp_supported and is_hip()):\n",
    )
    path.write_text(text, encoding="utf-8")

ep_moe_path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py")
ep_moe_text = ep_moe_path.read_text(encoding="utf-8")
if "Clypt ROCm patch: avoid importing Quark MoE schemes by default" not in ep_moe_text:
    ep_moe_text = ep_moe_text.replace("import logging\n", "import logging\nimport os\n")
    ep_moe_text = ep_moe_text.replace(
        "from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE\n",
        "# Clypt ROCm patch: avoid importing Quark MoE schemes by default.\n"
        "# Qwen/Qwen3.6-35B-A3B is not a Quark checkpoint, and the upstream\n"
        "# Quark import path pulls an incompatible bundled AITER module on this\n"
        "# ROCm image before the Qwen3.5 MoE implementation can register.\n"
        "if os.getenv(\"SGLANG_ENABLE_QUARK_QUANTIZATION\", \"0\").lower() in {\"1\", \"true\", \"yes\", \"on\"}:\n"
        "    from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE\n"
        "else:\n"
        "    class QuarkW4A4MXFp4MoE:  # type: ignore[no-redef]\n"
        "        pass\n",
    )
    ep_moe_path.write_text(ep_moe_text, encoding="utf-8")
PY
