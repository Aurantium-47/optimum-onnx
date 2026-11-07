import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from transformers import GenerationConfig, GPT2Config

from optimum.onnxruntime import ORTModelForCausalLM


@pytest.fixture
def model(tmp_path):
    # Position-sensitive logits, and real growing key/value cache tensors.
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "sequence"])
        for n in ["input_ids", "position_ids"]
    ]
    inputs += [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, ["batch", 1, "past", 1])
        for n in ["past_key_values.0.key", "past_key_values.0.value"]
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", 16])]
    outputs += [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, ["batch", 1, "total", 1])
        for n in ["present.0.key", "present.0.value"]
    ]
    constants = [
        numpy_helper.from_array(np.array([1, 3], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array(16, dtype=np.int64), "depth"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.float32), "values"),
    ]
    nodes = [
        helper.make_node("Add", ["input_ids", "position_ids"], ["indices"]),
        helper.make_node("OneHot", ["indices", "depth", "values"], ["logits"], axis=-1),
        helper.make_node("Cast", ["input_ids"], ["ids_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_float", "axes"], ["new_cache"]),
    ]
    for kind in ["key", "value"]:
        nodes.append(
            helper.make_node("Concat", [f"past_key_values.0.{kind}", "new_cache"], [f"present.0.{kind}"], axis=2)
        )
    graph = helper.make_graph(nodes, "position_sensitive_decoder", inputs, outputs, constants)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(proto)
    path = tmp_path / "model.onnx"
    onnx.save(proto, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    config = GPT2Config(n_layer=1, n_head=1, n_embd=1, vocab_size=16, use_cache=True)
    return ORTModelForCausalLM(
        config=config,
        session=session,
        use_io_binding=False,
        generation_config=GenerationConfig.from_model_config(config),
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("suffix", [1, 2, 3])
def test_cached_multiple_tokens_match_explicit_positions(model, batch, suffix):
    past = 3
    ids = torch.ones((batch, past + suffix), dtype=torch.long)
    mask = torch.ones_like(ids)
    if batch == 2:
        mask[1, 0] = 0
    cache = ((torch.zeros(batch, 1, past, 1), torch.zeros(batch, 1, past, 1)),)
    prepared = model.prepare_inputs_for_generation(
        ids,
        attention_mask=mask,
        past_key_values=cache,
        cache_position=torch.arange(past, past + suffix),
        use_cache=True,
    )
    expected = mask.cumsum(-1) - 1
    expected.masked_fill_(mask == 0, 1)
    expected = expected[:, -suffix:]
    expected_outputs = model(input_ids=ids[:, -suffix:], position_ids=expected, past_key_values=cache, use_cache=True)
    actual = model(**prepared)
    torch.testing.assert_close(actual.logits, expected_outputs.logits)
    torch.testing.assert_close(actual.past_key_values[0][0], expected_outputs.past_key_values[0][0])
    assert prepared["position_ids"].shape == prepared["input_ids"].shape


@pytest.mark.parametrize("past", [0, None])
def test_first_forward_keeps_all_positions(model, past):
    ids = torch.ones((1, 3), dtype=torch.long)
    mask = torch.ones_like(ids)
    cache = None if past is None else ((torch.zeros(1, 1, 0, 1), torch.zeros(1, 1, 0, 1)),)
    prepared = model.prepare_inputs_for_generation(
        ids, attention_mask=mask, past_key_values=cache, cache_position=torch.arange(3), use_cache=True
    )
    assert prepared["position_ids"].tolist() == [[0, 1, 2]]
    assert model(**prepared).logits.argmax(-1).tolist() == [[1, 2, 3]]


def test_explicit_positions_preserved(model):
    ids = torch.ones((1, 5), dtype=torch.long)
    positions = torch.tensor([[3, 4]])
    cache = ((torch.zeros(1, 1, 3, 1), torch.zeros(1, 1, 3, 1)),)
    prepared = model.prepare_inputs_for_generation(
        ids, position_ids=positions, past_key_values=cache, cache_position=torch.arange(3, 5), use_cache=True
    )
    torch.testing.assert_close(prepared["position_ids"], positions)
    assert model(**prepared).logits.argmax(-1).tolist() == [[4, 5]]
