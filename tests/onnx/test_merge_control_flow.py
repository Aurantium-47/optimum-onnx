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
import itertools
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from optimum.onnx.graph_transformations import merge_decoders, remove_duplicate_weights
from optimum.onnx.transformations_utils import _replace_input_names


def _graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _graphs(subgraph)


def _value(name, dtype=TensorProto.FLOAT, shape=(1, 2)):
    return helper.make_tensor_value_info(name, dtype, shape)


def _weight(value, name="weight"):
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def _if_branch(name, negate=False, collision=None):
    initializers = []
    sparse_initializers = []
    nodes = []
    weight_name = "weight"
    if collision is not None:
        # These sibling-local definitions are legal before merging, and must not
        # become the name of an initializer lifted into the enclosing graph.
        local = _weight([[2, 0], [0, 2]], "weight_merged_0")
        if collision == "initializer":
            initializers.append(local)
        elif collision == "sparse_initializer":
            # Even an unused sparse declaration reserves its tensor name.
            sparse_initializers.append(
                helper.make_sparse_tensor(
                    _weight([2, 2], "weight_merged_0"),
                    numpy_helper.from_array(np.array([[0, 0], [1, 1]], dtype=np.int64)),
                    [2, 2],
                )
            )
        else:
            nodes.append(helper.make_node("Constant", [], [local.name], value=local))
        if collision != "sparse_initializer":
            nodes.append(helper.make_node("Add", ["weight", local.name], ["combined_weight"]))
            weight_name = "combined_weight"
    nodes.append(helper.make_node("MatMul", ["x", weight_name], ["product"]))
    nodes.append(helper.make_node("Neg" if negate else "Identity", ["product"], ["branch_output"]))
    graph = helper.make_graph(nodes, name, [], [_value("branch_output")], initializer=initializers)
    graph.sparse_initializer.extend(sparse_initializers)
    # A captured value may also have type/shape metadata in a subgraph.
    graph.value_info.append(_value("weight", shape=(2, 2)))
    return graph


def _decoder(kind, weight, collision=None):
    inputs = [_value("x")]
    initializers = [_weight(weight)]
    if kind in {"if", "nested_if"}:
        inputs.append(_value("condition", TensorProto.BOOL, ()))
        then_branch = _if_branch("then", collision=collision)
        else_branch = _if_branch("else", negate=True, collision=collision)
        if kind == "nested_if":
            inputs.append(_value("inner_condition", TensorProto.BOOL, ()))
            inner = helper.make_node(
                "If", ["inner_condition"], ["nested_output"], then_branch=then_branch, else_branch=else_branch
            )
            then_branch = helper.make_graph([inner], "nested", [], [_value("nested_output")])
        nodes = [helper.make_node("If", ["condition"], ["y"], then_branch=then_branch, else_branch=else_branch)]
    elif kind == "loop":
        initializers.extend(
            [
                numpy_helper.from_array(np.array(2, dtype=np.int64), "trip_count"),
                numpy_helper.from_array(np.array(True), "condition"),
            ]
        )
        state_name = "weight_merged_0" if collision == "input" else "state"
        body = helper.make_graph(
            [
                helper.make_node("Identity", ["loop_condition"], ["next_condition"]),
                helper.make_node("MatMul", [state_name, "weight"], ["next_state"]),
            ],
            "loop_body",
            [
                _value("iteration", TensorProto.INT64, ()),
                _value("loop_condition", TensorProto.BOOL, ()),
                _value(state_name),
            ],
            [_value("next_condition", TensorProto.BOOL, ()), _value("next_state")],
        )
        nodes = [helper.make_node("Loop", ["trip_count", "condition", "x"], ["y"], body=body)]
    elif kind == "scan":
        inputs = [_value("x", shape=(3, 2))]
        body = helper.make_graph(
            [helper.make_node("MatMul", ["row", "weight"], ["next_row"])],
            "scan_body",
            [_value("row", shape=(2,))],
            [_value("next_row", shape=(2,))],
        )
        nodes = [helper.make_node("Scan", ["x"], ["y"], body=body, num_scan_inputs=1)]
    else:
        nodes = [helper.make_node("MatMul", ["x", "weight"], ["y"])]
    output_shape = (3, 2) if kind == "scan" else (1, 2)
    return helper.make_model(
        helper.make_graph(nodes, "decoder", inputs, [_value("y", shape=output_shape)], initializer=initializers),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )


def _session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


def _check_merge(tmp_path, kind, different_weights=False, collision=None):
    first_weight = [[1, 2], [3, 4]]
    second_weight = [[5, 6], [7, 8]] if different_weights else first_weight
    models = [_decoder(kind, first_weight, collision), _decoder(kind, second_weight, collision)]
    sessions = []
    for model in models:
        onnx.checker.check_model(model, full_check=True)
        sessions.append(_session(model))
    merged = merge_decoders(*models, save_path=tmp_path / "merged.onnx")
    onnx.checker.check_model(merged, full_check=True)
    merged_session = _session(merged)
    assert len(merged.graph.initializer) == (2 if different_weights else 1)
    assert [output.name for output in merged.graph.output] == ["y"]
    for graph in _graphs(merged.graph):
        assert "weight" not in {value_info.name for value_info in graph.value_info}
    data = [[2, 3], [4, 5], [6, 7]] if kind == "scan" else [[2, 3]]
    for cache, condition, inner_condition in itertools.product([False, True], repeat=3):
        feeds = {"x": np.array(data, dtype=np.float32)}
        if kind in {"if", "nested_if"}:
            feeds["condition"] = np.array(condition)
        if kind == "nested_if":
            feeds["inner_condition"] = np.array(inner_condition)
        expected = sessions[int(cache)].run(None, feeds)[0]
        actual = merged_session.run(None, {**feeds, "use_cache_branch": np.array([cache])})[0]
        np.testing.assert_array_equal(actual, expected)
    return merged


@pytest.mark.parametrize("kind", ["root", "if", "nested_if", "loop", "scan"])
@pytest.mark.parametrize("different_weights", [False, True])
def test_merge_decoders_control_flow_captures(tmp_path: Path, kind, different_weights):
    _check_merge(tmp_path, kind, different_weights=different_weights)


@pytest.mark.parametrize(
    "kind,collision",
    [("if", "initializer"), ("if", "sparse_initializer"), ("if", "node_output"), ("loop", "input")],
)
def test_merge_decoders_avoids_subgraph_name_collisions(tmp_path: Path, kind, collision):
    merged = _check_merge(tmp_path, kind, different_weights=True, collision=collision)
    assert "weight_merged_0" not in {initializer.name for initializer in merged.graph.initializer}


def test_replace_input_names_visits_graph_list_attributes():
    # GRAPHS is available to custom operators, but no standard operator with
    # that attribute type can be executed by the CPU runtime.
    node = helper.make_node("GraphCollection", [], [], domain="test", branches=[_if_branch("a"), _if_branch("b")])
    model = helper.make_model(helper.make_graph([node], "root", [], []))
    _replace_input_names([model], {("weight", 0): "shared_weight"})
    graphs = list(_graphs(model.graph))
    assert len(graphs) == 3
    for graph in graphs[1:]:
        assert list(graph.node[0].input) == ["x", "shared_weight"]
        assert [value_info.name for value_info in graph.value_info] == ["shared_weight"]


@pytest.mark.parametrize("kind", ["if", "nested_if"])
def test_remove_duplicate_weights_preserves_subgraph_captures(kind):
    weight = [[1, 2], [3, 4]]
    model = _decoder(kind, weight)
    model.graph.initializer.append(_weight(weight, "duplicate_weight"))
    # Both duplicate names are captured, so the regression does not depend on
    # which initializer the deduplication chooses as its representative.
    first_branch = next(graph for graph in _graphs(model.graph) if graph.name == "else")
    first_branch.node[0].input[1] = "duplicate_weight"
    first_branch.value_info[0].name = "duplicate_weight"
    onnx.checker.check_model(model, full_check=True)
    original_session = _session(model)
    deduplicated = remove_duplicate_weights(model)
    assert len(model.graph.initializer) == 2
    assert len(deduplicated.graph.initializer) == 1
    onnx.checker.check_model(deduplicated, full_check=True)
    deduplicated_session = _session(deduplicated)
    for condition, inner_condition in itertools.product([False, True], repeat=2):
        feeds = {"x": np.array([[2, 3]], dtype=np.float32), "condition": np.array(condition)}
        if kind == "nested_if":
            feeds["inner_condition"] = np.array(inner_condition)
        np.testing.assert_array_equal(deduplicated_session.run(None, feeds)[0], original_session.run(None, feeds)[0])


class _ConditionalDecoder(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if bool(condition):
            return torch.matmul(x, self.weight)
        return -torch.matmul(x, self.weight)


def test_merge_torchscript_conditional_decoders(tmp_path: Path):
    x = torch.tensor([[2.0, 3.0]])
    torch_models = [_ConditionalDecoder([[1, 2], [3, 4]]).eval(), _ConditionalDecoder([[5, 6], [7, 8]]).eval()]
    models = []
    for index, torch_model in enumerate(torch_models):
        path = tmp_path / f"decoder_{index}.onnx"
        torch.onnx.export(
            torch.jit.script(torch_model),
            (x, torch.tensor(True)),
            path,
            input_names=["x", "condition"],
            output_names=["y"],
            opset_version=17,
            dynamo=False,
        )
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        assert any(node.op_type == "If" for node in model.graph.node)
        models.append(model)
    original_sessions = [_session(model) for model in models]
    merged = merge_decoders(*models, save_path=tmp_path / "merged.onnx")
    onnx.checker.check_model(merged, full_check=True)
    merged_session = _session(merged)
    for cache, condition in itertools.product([False, True], repeat=2):
        feeds = {"x": x.numpy(), "condition": np.array(condition)}
        with torch.no_grad():
            expected = torch_models[int(cache)](x, torch.tensor(condition)).numpy()
        np.testing.assert_array_equal(original_sessions[int(cache)].run(None, feeds)[0], expected)
        np.testing.assert_array_equal(
            merged_session.run(None, {**feeds, "use_cache_branch": np.array([cache])})[0], expected
        )
