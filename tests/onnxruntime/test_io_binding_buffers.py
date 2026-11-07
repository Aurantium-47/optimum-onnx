# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper

from optimum.onnxruntime.base import ORTSessionMixin


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reuse_buffer", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "offset_contiguous", "transposed", "strided", "channels_last"])
def test_reused_output_buffer_preserves_logical_tensor_order(tmp_path, layout, reuse_buffer, dtype):
    ort.disable_telemetry_events()
    if layout == "channels_last":
        sample = torch.arange(48, dtype=dtype).reshape(2, 3, 2, 4).to(memory_format=torch.channels_last)
    elif layout == "offset_contiguous":
        sample = torch.arange(15, dtype=dtype)[3:].reshape(2, 6)
    elif layout == "transposed":
        sample = torch.arange(12, dtype=dtype).reshape(2, 6).transpose(0, 1)
    elif layout == "strided":
        sample = torch.arange(26, dtype=dtype).reshape(2, 13)[:, 1::2]
    else:
        sample = torch.arange(12, dtype=dtype).reshape(2, 6)
    original = sample.clone()
    expected = original + 1
    tensor_type = TensorProto.FLOAT16 if dtype == torch.float16 else TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node("Add", ["sample", "one"], ["out_sample"])],
        "reused_output_buffer",
        [helper.make_tensor_value_info("sample", tensor_type, list(sample.shape))],
        [helper.make_tensor_value_info("out_sample", tensor_type, list(sample.shape))],
        [helper.make_tensor("one", tensor_type, [], [1.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    wrapper = ORTSessionMixin()
    wrapper.initialize_ort_attributes(session, use_io_binding=True)
    unbound = session.run(None, {"sample": sample.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(unbound), expected)

    model_inputs = {"sample": sample}
    shapes, buffers = wrapper._prepare_io_binding(
        model_inputs,
        known_output_buffers={"out_sample": sample} if reuse_buffer else None,
        known_output_shapes={"out_sample": tuple(sample.shape)},
    )
    session.run_with_iobinding(wrapper._io_binding)
    actual = buffers["out_sample"].view(shapes["out_sample"])
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == sample.dtype
    assert actual.device == sample.device
    assert actual.shape == sample.shape
    assert actual.is_contiguous()
    if reuse_buffer and sample.is_contiguous():
        assert buffers["out_sample"] is sample
    else:
        assert buffers["out_sample"].data_ptr() != sample.data_ptr()
        torch.testing.assert_close(sample, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("use_io_binding", [False, True])
@pytest.mark.parametrize("reuse_buffer", [False, True])
def test_unet_output_buffer_matches_declared_dtype(tmp_path, dtype, transposed, use_io_binding, reuse_buffer):
    pytest.importorskip("diffusers")
    from optimum.onnxruntime.modeling_diffusion import ORTUnet

    ort.disable_telemetry_events()
    backing = torch.full((20,), -99.0, dtype=dtype)
    sample = backing[4:16].reshape(1, 1, 3, 4)
    sample.copy_(torch.arange(12).reshape(sample.shape))
    if transposed:
        sample = sample.transpose(-1, -2)
    original = sample.clone()
    guards = backing.clone()
    expected = original.to(torch.float16) + 1
    # The graph only writes float16, so even the failing float32 case has enough storage.
    graph = helper.make_graph(
        [helper.make_node("Add", ["sample", "one"], ["out_sample"])],
        "unet_reused_output_buffer",
        [
            helper.make_tensor_value_info("sample", TensorProto.FLOAT16, list(sample.shape)),
            helper.make_tensor_value_info("timestep", TensorProto.FLOAT16, []),
            helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT16, [1, 2, 3]),
        ],
        [helper.make_tensor_value_info("out_sample", TensorProto.FLOAT16, list(sample.shape))],
        [helper.make_tensor("one", TensorProto.FLOAT16, [], [1.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    (tmp_path / "config.json").write_text('{"time_cond_proj_dim": null}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    # Forward only uses the parent's class name to decide whether to reuse the sample.
    parent_name = "StableDiffusionPipeline" if reuse_buffer else "LatentConsistencyModelPipeline"
    parent = type(parent_name, (), {})()
    unet = ORTUnet(session, parent, use_io_binding=use_io_binding)
    actual = unet.forward(sample, torch.tensor(1.0, dtype=dtype), torch.ones(1, 2, 3, dtype=dtype)).sample

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float16
    assert actual.device == sample.device
    assert actual.is_contiguous()
    torch.testing.assert_close(backing[:4], guards[:4])
    torch.testing.assert_close(backing[16:], guards[16:])
    if use_io_binding and reuse_buffer and dtype == torch.float16 and not transposed:
        assert actual.data_ptr() == sample.data_ptr()
    else:
        assert actual.data_ptr() != sample.data_ptr()
        torch.testing.assert_close(sample, original)
