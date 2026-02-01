# Repository Guidelines

## Project Structure & Module Organization
- `main_vlm.py` is the primary entry point for the VLM-driven grasp pipeline.
- Core Python modules live at repo root: `vlm_process.py`, `grasp_process_optimized.py`, and related helpers.
- `manipulator_grasp/` provides robot arm simulation/control, motion planning, and bundled assets (UR5e, iiwa, Robotiq).
- `graspnet-baseline/` is a vendor subtree for GraspNet baseline models and custom CUDA ops (`pointnet2/`, `knn/`).
- `logs/log_rs/` stores pretrained checkpoints such as `checkpoint-rs.tar`.

## Build, Test, and Development Commands
- Create environment: `conda create -n vlm_graspnet python=3.11` then `conda activate vlm_graspnet`.
- Install GraspNet dependencies: `pip install -r graspnet-baseline/requirements.txt`.
- Build CUDA extensions:
  - `cd graspnet-baseline/pointnet2 && python setup.py install`
  - `cd graspnet-baseline/knn && python setup.py install`
- Install GraspNet API: `cd graspnet-baseline/graspnetAPI && pip install .`
- Run the main pipeline: `python main_vlm.py`.

## Coding Style & Naming Conventions
- Python code; follow PEP 8 with 4-space indentation and `snake_case` for functions/variables.
- Modules/files are lowercase with underscores (e.g., `grasp_process_optimized.py`).
- No formatter or linter is configured; keep diffs minimal and readable.

## Testing Guidelines
- No automated test suite is present. If you add tests, prefer `pytest` and place them under a new `tests/` directory (e.g., `tests/test_vlm_process.py`).
- For now, validate changes by running `python main_vlm.py` with a known checkpoint in `logs/log_rs/`.

## Commit & Pull Request Guidelines
- Git history is minimal (single commit with a timestamp-style message). There is no established commit convention.
- Use short, descriptive commit messages (e.g., `Fix grasp pose filtering`), and include context in the PR description.
- If a change affects model outputs or assets, attach before/after notes or screenshots in the PR.

## Notes for Contributors
- See `README_vlm_grasping.md` for algorithm versioning and debugging history; it documents known pitfalls (e.g., proxy settings and camera frame alignment).
- Avoid modifying vendor subtrees (`graspnet-baseline/`, `manipulator_grasp/`) unless the change is intentional and documented.
