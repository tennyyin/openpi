"""Tests for the pure rate/alignment logic in convert_wm_demos_to_lerobot.

WM demos come off the world model at the AR BLOCK rate (2.5 Hz for the shipped 10 fps
tri_bike checkpoint, 1.25 Hz for a 5 fps one) but must be written at a FIXED 10 Hz so
they mix with the real TRI demos. These cover that resample, the meta-driven rate
resolution behind it, and the NaN-state segmentation -- all without touching lerobot or
a GPU. Run with the openpi uv env: `uv run pytest examples/bike_rotor/`."""

import importlib.util
import json
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "convert_wm_demos", os.path.join(_HERE, "convert_wm_demos_to_lerobot.py"))
cwd = importlib.util.module_from_spec(_spec)
# Executing the module imports lerobot at top level; if unavailable, skip the file.
try:
    _spec.loader.exec_module(cwd)
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAVE, reason="lerobot deps unavailable")


def test_block_frame_index_is_block_center():
    # rgb_per_block=4: block 0 -> frame 2, block 1 -> frame 6, block 2 -> 10 ...
    assert cwd.block_frame_index(0, 4, 100) == 2
    assert cwd.block_frame_index(1, 4, 100) == 6
    assert cwd.block_frame_index(3, 4, 100) == 14


def test_block_frame_index_clamps_to_available_frames():
    # if the tail was truncated (--record-max-frames), the center clamps to the last frame
    assert cwd.block_frame_index(5, 4, 12) == 11        # would be 22, clamped to 11
    assert cwd.block_frame_index(0, 4, 1) == 0


def test_align_blocks_full():
    # 3 blocks, 4 frames each, all 12 present -> centers [2, 6, 10]
    assert cwd.align_blocks(3, 4, 12) == [2, 6, 10]


def test_align_blocks_drops_truncated_tail():
    # only 9 frames written for 3 blocks*4: block 2 starts at frame 8 (present) -> kept
    # (its center clamps into range); a 4th block would start at 12 (absent) -> dropped.
    assert cwd.align_blocks(4, 4, 9) == [2, 6, cwd.block_frame_index(2, 4, 9)]
    assert cwd.align_blocks(4, 4, 9)[-1] == 8           # block 2 center 10 -> clamp to 8


def test_align_blocks_empty_when_no_frames():
    assert cwd.align_blocks(5, 4, 0) == []


def test_view_and_dim_contract():
    # the converter targets the exact bike_rotor schema
    assert set(cwd.VIEW_TO_SLOT.values()) == {
        "observation.images.base", "observation.images.left_wrist", "observation.images.right_wrist"}
    assert cwd.STATE_DIM == 16 and cwd.ACTION_DIM == 20
    assert (cwd.RESIZE_H, cwd.RESIZE_W) == (224, 224)
    # rot6d spans of the 20-d [R_xyz, R_rot6d, L_xyz, L_rot6d, gripR, gripL] vector
    assert cwd.ACTION_ROT6D_SLICES == ((3, 9), (12, 18))
    assert cwd.RECORD_TARGET_HZ == 10.0


# --- rate resolution from meta.json -----------------------------------------
def _meta(**kw):
    base = {"data_fps": 10.0, "block_hz": 2.5, "frames_per_block": 1,
            "rgb_frames_per_block": 4, "record_target_hz": 10.0}
    base.update(kw)
    return base


def test_demo_rates_reads_the_10fps_checkpoint():
    assert cwd.demo_rates(_meta()) == (10.0, 2.5, 4)


def test_demo_rates_reads_the_5fps_checkpoint():
    # THE case this work exists for: a 5 fps root gives 1.25 Hz blocks, same dims and
    # same stats filenames as the 10 fps one -- only meta distinguishes them.
    assert cwd.demo_rates(_meta(data_fps=5.0, block_hz=1.25)) == (5.0, 1.25, 4)


def test_demo_rates_rejects_inconsistent_meta():
    # data_fps/rgb_per_block and block_hz come from the same resolver upstream; if they
    # disagree, every timestamp below is wrong with no other symptom.
    with pytest.raises(ValueError, match="disagrees"):
        cwd.demo_rates(_meta(block_hz=10.0))
    # ...unless overridden deliberately
    assert cwd.demo_rates(_meta(block_hz=10.0), block_hz_override=10.0) == (10.0, 10.0, 4)


def test_demo_rates_needs_a_video_rate():
    m = _meta()
    del m["data_fps"], m["block_hz"]
    with pytest.raises(ValueError, match="no data_fps"):
        cwd.demo_rates(m)
    # legacy demos fall back to meta['fps'] (with a warning) rather than failing
    m["fps"] = 10
    assert cwd.demo_rates(m) == (10.0, 2.5, 4)


def test_demo_rates_derives_block_hz_when_absent():
    m = _meta()
    del m["block_hz"]
    assert cwd.demo_rates(m) == (10.0, 2.5, 4)


# --- the resample plan -------------------------------------------------------
def test_resample_at_block_rate_reproduces_one_row_per_block():
    # target_hz == block_hz must degrade EXACTLY to the old behaviour (block centres),
    # so the resample cannot change results for anyone converting at the block rate.
    coords, frames = cwd.resample_indices(3, 4, 12, block_hz=2.5, data_fps=10.0,
                                          target_hz=2.5)
    assert coords.tolist() == [0.0, 1.0, 2.0]
    assert frames.tolist() == cwd.align_blocks(3, 4, 12) == [2, 6, 10]


def test_resample_10fps_root_upsamples_blocks_4x():
    # 10 fps video, 2.5 Hz blocks -> 10 Hz output: 4 rows per block, advancing 1 frame
    # and 0.25 block each. Frames start at the first block's centre (2).
    coords, frames = cwd.resample_indices(3, 4, 12, block_hz=2.5, data_fps=10.0,
                                          target_hz=10.0)
    assert coords.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert frames.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # neither stream is extrapolated: last coord is the last block, last frame exists
    assert coords[-1] == 2.0 and frames[-1] <= 11


def test_resample_5fps_root_lands_on_the_same_10hz_grid():
    # 5 fps video, 1.25 Hz blocks -> 10 Hz: 8 rows per block, 0.5 frame per row. The
    # SAME wall-clock duration as the 10 fps case above yields the SAME row count, which
    # is the whole point -- a 5 Hz model's demos are mixable with a 10 Hz model's.
    coords, frames = cwd.resample_indices(3, 4, 12, block_hz=1.25, data_fps=5.0,
                                          target_hz=10.0)
    assert coords.tolist() == [i * 0.125 for i in range(17)]
    assert frames.tolist() == [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    assert coords[-1] == 2.0


def test_resample_stops_at_truncated_video():
    # --record-max-frames caps the video while keeping every action; rows past the last
    # written frame would all repeat it, so they are not emitted.
    coords, frames = cwd.resample_indices(10, 4, 8, block_hz=2.5, data_fps=10.0,
                                          target_hz=10.0)
    assert frames.max() <= 7
    assert coords[-1] < 9.0                      # did not run to the last action block
    assert cwd.resample_indices(5, 4, 0, 2.5, 10.0, 10.0)[0].size == 0
    assert cwd.resample_indices(0, 4, 12, 2.5, 10.0, 10.0)[0].size == 0


# --- low-dim resampling ------------------------------------------------------
def test_resample_lowdim_interpolates_and_holds():
    arr = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float32)
    coords = np.array([0.0, 0.5, 1.25, 2.0])
    out = cwd.resample_lowdim(arr, coords, mode="interp")
    assert np.allclose(out, [[0.0, 10.0], [0.5, 15.0], [1.25, 22.5], [2.0, 30.0]])
    held = cwd.resample_lowdim(arr, coords, mode="hold")
    assert np.allclose(held, [[0.0, 10.0], [0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    assert out.dtype == np.float32 and held.dtype == np.float32
    # past-the-end coords clamp instead of extrapolating
    assert np.allclose(cwd.resample_lowdim(arr, np.array([5.0]))[0], [2.0, 30.0])


def _rot6d(mat):
    return np.concatenate([mat[:, 0], mat[:, 1]])


def test_resample_lowdim_slerps_rot6d_on_so3():
    # The reason rot6d needs its own path: a component-wise lerp of two rot6d rows is not
    # orthonormal, and Gram-Schmidt then reads it as a DIFFERENT rotation than either
    # endpoint. Assert the interpolated frames are proper rotations AND that the midpoint
    # is the true half-way rotation (rotvec exactly half of the endpoint-to-endpoint one).
    from scipy.spatial.transform import Rotation as R

    R0 = R.from_euler("xyz", [0.1, -0.2, 0.3])
    R1 = R0 * R.from_rotvec([0.0, 0.9, 0.0])         # 0.9 rad about the tool y axis
    arr = np.zeros((2, 20), dtype=np.float32)
    arr[0, 3:9] = _rot6d(R0.as_matrix())
    arr[1, 3:9] = _rot6d(R1.as_matrix())
    arr[0, 12:18] = _rot6d(R0.as_matrix())          # left arm static
    arr[1, 12:18] = _rot6d(R0.as_matrix())
    arr[0, 0:3], arr[1, 0:3] = [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]

    out = cwd.resample_lowdim(arr, np.array([0.0, 0.5, 1.0]),
                              rot_slices=cwd.ACTION_ROT6D_SLICES)

    def mat(r6):
        b1 = r6[0:3] / np.linalg.norm(r6[0:3])
        b2 = r6[3:6] - np.dot(b1, r6[3:6]) * b1
        b2 = b2 / np.linalg.norm(b2)
        return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)

    for row in out:
        for s, e in cwd.ACTION_ROT6D_SLICES:
            M = mat(row[s:e].astype(np.float64))
            assert np.allclose(M.T @ M, np.eye(3), atol=1e-5)      # orthonormal
            assert np.isclose(np.linalg.det(M), 1.0, atol=1e-5)    # right-handed
    # endpoints preserved, midpoint is the true half rotation
    assert np.allclose(mat(out[0, 3:9].astype(np.float64)), R0.as_matrix(), atol=1e-5)
    assert np.allclose(mat(out[2, 3:9].astype(np.float64)), R1.as_matrix(), atol=1e-5)
    half = (R0.inv() * R.from_matrix(mat(out[1, 3:9].astype(np.float64)))).as_rotvec()
    assert np.allclose(half, [0.0, 0.45, 0.0], atol=1e-5), half
    # translation still lerps, and the untouched arm stays put
    assert np.allclose(out[1, 0:3], [0.5, 1.0, 1.5], atol=1e-5)
    assert np.allclose(out[1, 12:18], arr[0, 12:18], atol=1e-5)


def test_slerp_handles_identical_and_antipodal_quaternions():
    # theta -> 0 must not divide by ~0, and the double cover must take the SHORT arc.
    from scipy.spatial.transform import Rotation as R

    R0 = R.from_euler("xyz", [0.4, 0.1, -0.2])
    same = np.tile(_rot6d(R0.as_matrix()), (3, 1))
    out = cwd._slerp_rot6d(same, same.copy(), np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(out).all()
    assert np.allclose(out, same, atol=1e-6)
    # ~pi apart: still a valid rotation at the midpoint
    R1 = R0 * R.from_rotvec([0.0, 0.0, np.pi - 1e-3])
    out2 = cwd._slerp_rot6d(_rot6d(R0.as_matrix())[None], _rot6d(R1.as_matrix())[None],
                            np.array([0.5]))
    assert np.isfinite(out2).all()
    assert np.isclose(np.linalg.norm(out2[0, 0:3]), 1.0, atol=1e-5)


# --- NaN-state segmentation --------------------------------------------------
def test_valid_segments_trims_edges_and_splits_holes():
    # NaN blocks (aux state head missed a block) cannot be interpolated THROUGH: the
    # result is either NaN or a plausible number bridging a gap nobody commanded.
    assert cwd.valid_segments(np.array([1, 1, 1, 1], bool)) == [(0, 4)]
    assert cwd.valid_segments(np.array([0, 1, 1, 1, 0], bool)) == [(1, 4)]
    assert cwd.valid_segments(np.array([1, 1, 0, 1, 1], bool)) == [(0, 2), (3, 5)]
    # runs shorter than min_blocks are dropped, not emitted as 1-row episodes
    assert cwd.valid_segments(np.array([1, 0, 1, 1], bool), min_blocks=2) == [(2, 4)]
    assert cwd.valid_segments(np.zeros(4, bool)) == []


def test_plan_demo_drops_nan_rows_and_hits_the_target_rate():
    class _A:
        target_fps, resample, block_hz, min_frames, min_blocks = 10, "interp", 0.0, 2, 2

    n_blocks, rgb = 6, 4
    state = np.tile(np.arange(n_blocks, dtype=np.float32)[:, None], (1, 16))
    actions = np.zeros((n_blocks, 20), dtype=np.float32)
    actions[:, 3:9] = actions[:, 12:18] = np.array([1, 0, 0, 0, 1, 0], np.float32)
    valid = np.ones(n_blocks, bool)
    valid[[0, 3]] = False                     # a leading trim and an interior hole
    state[~valid] = np.nan

    eps = cwd.plan_demo(state, actions, valid, n_blocks * rgb, _meta(), _A())
    # blocks 1-2 and 4-5 -> two episodes, never one with a jump across block 3
    assert len(eps) == 2
    for seg_state, seg_actions, frames in eps:
        assert np.isfinite(seg_state).all() and np.isfinite(seg_actions).all()
        assert seg_state.shape[1] == 16 and seg_actions.shape[1] == 20
        # 2 blocks @ 2.5 Hz -> 5 rows @ 10 Hz (0, .25, .5, .75, 1.0 blocks)
        assert seg_state.shape[0] == 5 == frames.shape[0]
        assert frames.max() < n_blocks * rgb
    # the second episode's frames come AFTER the first's (indices offset by the trim)
    assert eps[1][2].min() > eps[0][2].max()
    # and its state values are the later blocks' (4..5), proving the offset is applied
    assert np.isclose(eps[1][0][0, 0], 4.0)


def test_plan_demo_skips_a_demo_with_no_usable_segment():
    class _A:
        target_fps, resample, block_hz, min_frames, min_blocks = 10, "interp", 0.0, 2, 2

    state = np.full((3, 16), np.nan, np.float32)
    actions = np.zeros((3, 20), np.float32)
    assert cwd.plan_demo(state, actions, np.zeros(3, bool), 12, _meta(), _A()) == []


# --- action mode: absolute cartesian only ------------------------------------
def _mode(**kw):
    base = {"action_space": "cartesian", "action_delta": False,
            "action_cond_mode": "cross_attn_aligned"}
    base.update(kw)
    return base


def test_absolute_cartesian_meta_is_accepted():
    cwd.require_absolute_cartesian(_mode())                 # must not raise
    for mode in cwd.REQUIRED_ACTION_MODE["action_cond_mode"]:
        cwd.require_absolute_cartesian(_mode(action_cond_mode=mode))


def test_joint_delta_and_actionless_demos_are_rejected():
    # Each of these converts CLEANLY to the right shapes and the wrong meaning, so the
    # only place it can be caught is here, against the mode the recorder wrote down.
    for bad, needle in ((_mode(action_space="joint_pos"), "action_space"),
                        (_mode(action_delta=True), "action_delta"),
                        (_mode(action_cond_mode="none"), "action_cond_mode")):
        with pytest.raises(ValueError, match=needle):
            cwd.require_absolute_cartesian(bad)


def test_missing_mode_fields_are_accepted_as_legacy():
    # Pre-gate demos have none of these keys and could only have been absolute cartesian
    # (no other mode had a working driving path), so absence must not fail the demo.
    cwd.require_absolute_cartesian({})
    cwd.require_absolute_cartesian(_mode(action_cond_mode=None))   # older recorder wrote null


def test_load_demo_checks_the_mode_before_decoding(tmp_path):
    # The gate must run before the mp4 decode: a wrong-mode demo is unconvertible, so
    # spending minutes of video decode on it (and only then failing) is wasted.
    d = tmp_path / "demo_0000"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(_meta(**_mode(action_delta=True))))
    with pytest.raises(ValueError, match="action_delta"):
        cwd.load_demo(str(d))       # no .npy / .mp4 files exist: must fail on the mode
    # ...and the whitelist spans exactly the three action-mode knobs open-world gates
    assert set(cwd.REQUIRED_ACTION_MODE) == {
        "action_space", "action_delta", "action_cond_mode"}
