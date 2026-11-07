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
