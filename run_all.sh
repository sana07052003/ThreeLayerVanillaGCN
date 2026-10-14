#!/usr/bin/env bash
set -e
python src/demo_fixed_weights.py > outputs/fixed_demo_output.txt
python tests/gradient_check.py    > outputs/gradient_check_output.txt
python tests/gradient_check_two_layer.py > outputs/gradient_check_two_layer_output.txt
python src/run_two_layer_example.py > outputs/two_layer_example_output.txt
python src/run_training.py        > outputs/training_output.txt      # small synthetic graph (supplementary)
python src/run_cora.py            | tee outputs/cora_output.txt        # main experiment (~12 min)
python src/make_figures.py
python src/make_cora_figures.py
python src/make_two_layer_figures.py
python src/build_two_layer_doc.py
python src/build_assignment_doc.py
python src/build_report.py
echo "Optional (needs PyTorch): python tests/test_torch_matches_numpy.py && python src/gcn_torch.py"
