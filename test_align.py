import numpy as np
import torch
import numba


# Original Cython-like implementation (translated to Python for testing)
def maximum_path_each_ref(path, value, t_x, t_y, max_neg_val=-1e9):
    index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]

    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


def maximum_path_ref(paths, values, t_xs, t_ys):
    b = values.shape[0]
    for i in range(b):
        maximum_path_each_ref(paths[i], values[i], t_xs[i], t_ys[i])


# Current Numba implementation
@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)
def maximum_path_jit(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or (y > 0 and value[y - 1, index] < value[y - 1, index - 1])
            ):
                index = index - 1


def test_alignment():
    # Create dummy data
    # neg_cent: [B, T_text, T_mel]
    # In VITS/Matcha, usually T_text is x, T_mel is y
    # But let's check the shapes in the code.
    # neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
    # neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (y**2), s_p_sq_r)
    # y is [B, D, T_mel], s_p_sq_r is [B, D, T_text]
    # So neg_cent is [B, T_mel, T_text]

    B, T_mel, T_text = 1, 5, 3
    neg_cent = np.random.randn(B, T_mel, T_text).astype(np.float32)
    t_mel = np.array([T_mel], dtype=np.int32)
    t_text = np.array([T_text], dtype=np.int32)

    # Test Reference
    path_ref = np.zeros((B, T_mel, T_text), dtype=np.int32)
    val_ref = neg_cent.copy()
    # Reference expects (paths, values, t_xs, t_ys) where t_xs is T_text, t_ys is T_mel
    # Wait, the Cython code says: maximum_path_each(path, value, t_x, t_y)
    # and value[x, y] = ...
    # So x is T_text, y is T_mel?
    # Let's re-read Cython: value[x, y-1]
    # If value is [T_text, T_mel], then x is text index, y is mel index.

    # In __init__.py:
    # t_t_max = mask.sum(1)[:, 0] -> T_mel?
    # t_s_max = mask.sum(2)[:, 0] -> T_text?
    # maximum_path_jit(path, neg_cent, t_t_max, t_s_max)

    # Let's try to match the Cython logic exactly.
    # Cython: value[x, y] = ... where x < t_x, y < t_y
    # So value is [t_x, t_y]

    T_X, T_Y = 3, 5  # text, mel
    neg_cent_ref = np.random.randn(B, T_X, T_Y).astype(np.float32)
    path_ref = np.zeros((B, T_X, T_Y), dtype=np.int32)
    val_ref = neg_cent_ref.copy()
    maximum_path_ref(
        path_ref,
        val_ref,
        np.array([T_X], dtype=np.int32),
        np.array([T_Y], dtype=np.int32),
    )

    # Test JIT
    # JIT signature: maximum_path_jit(paths, values, t_ys, t_xs)
    # It uses value[y, x]
    # So it expects values to be [B, T_Y, T_X]
    neg_cent_jit = neg_cent_ref.transpose(0, 2, 1).copy()
    path_jit = np.zeros((B, T_Y, T_X), dtype=np.int32)
    val_jit = neg_cent_jit.copy()
    maximum_path_jit(
        path_jit,
        val_jit,
        np.array([T_Y], dtype=np.int32),
        np.array([T_X], dtype=np.int32),
    )

    print("Reference Path:\n", path_ref[0])
    print("JIT Path (transposed):\n", path_jit[0].T)

    assert np.all(path_ref[0] == path_jit[0].T), "Paths do not match!"
    print("Test passed!")


if __name__ == "__main__":
    test_alignment()
