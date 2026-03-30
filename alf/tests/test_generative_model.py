import numpy as np
import pytest

from alf.generative_model import GenerativeModel


def test_generative_model_initialization_single_factor():
    """Test basic initialization and auto-computed attributes."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]  # (num_states, num_states, num_actions)
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    gm = GenerativeModel(A=A, B=B, C=C, D=D)

    assert gm.num_modalities == 1
    assert gm.num_factors == 1
    assert gm.num_obs == [2]
    assert gm.num_states == [2]
    assert gm.num_actions == [2]
    assert gm.T == 1

def test_generative_model_type_conversion():
    """Test that inputs are converted to float64 numpy arrays."""
    # Using python lists
    A = [[[0.8, 0.2], [0.2, 0.8]]]
    B = [[
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]]
    # Manually fix B shape for testing
    B_arr = np.array(B[0]).transpose(1, 2, 0)
    B = [B_arr.tolist()]

    C = [[1, -1]]
    D = [[0.5, 0.5]]

    gm = GenerativeModel(A=A, B=B, C=C, D=D)

    assert isinstance(gm.A[0], np.ndarray)
    assert gm.A[0].dtype == np.float64
    assert isinstance(gm.B[0], np.ndarray)
    assert gm.B[0].dtype == np.float64
    assert isinstance(gm.C[0], np.ndarray)
    assert gm.C[0].dtype == np.float64
    assert isinstance(gm.D[0], np.ndarray)
    assert gm.D[0].dtype == np.float64


def test_generative_model_policy_enumeration_single_factor_t1():
    """Test policy enumeration for a single factor model with T=1."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]  # 2 actions
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    gm = GenerativeModel(A=A, B=B, C=C, D=D, T=1)

    assert gm.num_policies == 2
    assert gm.policies.shape == (2, 1, 1)
    np.testing.assert_array_equal(gm.policies, np.array([[[0]], [[1]]]))

def test_generative_model_policy_enumeration_single_factor_t2():
    """Test policy enumeration for a single factor model with T=2."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]],
        [[0.5, 0.5], [0.5, 0.5]]
    ]).transpose(1, 2, 0)]  # 3 actions
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    gm = GenerativeModel(A=A, B=B, C=C, D=D, T=2)

    assert gm.num_policies == 9 # 3 actions ^ 2 timesteps
    assert gm.policies.shape == (9, 2, 1)

    expected_policies = np.array([
        [[0], [0]],
        [[0], [1]],
        [[0], [2]],
        [[1], [0]],
        [[1], [1]],
        [[1], [2]],
        [[2], [0]],
        [[2], [1]],
        [[2], [2]],
    ])
    np.testing.assert_array_equal(gm.policies, expected_policies)


def test_generative_model_policy_enumeration_multi_factor_t1():
    """Test policy enumeration for a multi factor model with T=1."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]

    # Factor 1: 2 states, 2 actions
    B1 = np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)

    # Factor 2: 2 states, 3 actions
    B2 = np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]],
        [[0.5, 0.5], [0.5, 0.5]]
    ]).transpose(1, 2, 0)

    B = [B1, B2]
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]

    gm = GenerativeModel(A=A, B=B, C=C, D=D, T=1)

    assert gm.num_factors == 2
    assert gm.num_policies == 6 # 2 actions * 3 actions
    assert gm.policies.shape == (6, 1, 2)

    expected_policies = np.array([
        [[0, 0]],
        [[0, 1]],
        [[0, 2]],
        [[1, 0]],
        [[1, 1]],
        [[1, 2]],
    ])
    np.testing.assert_array_equal(gm.policies, expected_policies)

def test_generative_model_policy_enumeration_multi_factor_t2():
    """Test policy enumeration for a multi factor model with T=2."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]

    # Factor 1: 2 states, 2 actions
    B1 = np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)

    # Factor 2: 2 states, 2 actions
    B2 = np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]],
    ]).transpose(1, 2, 0)

    B = [B1, B2]
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]

    gm = GenerativeModel(A=A, B=B, C=C, D=D, T=2)

    assert gm.num_factors == 2
    # 2*2 action combinations = 4 joint actions
    # T=2, so 4^2 = 16 policies
    assert gm.num_policies == 16
    assert gm.policies.shape == (16, 2, 2)

    # Verify the first few and last few policies
    expected_first = np.array([[0, 0], [0, 0]])
    expected_second = np.array([[0, 0], [0, 1]])
    expected_last = np.array([[1, 1], [1, 1]])

    np.testing.assert_array_equal(gm.policies[0], expected_first)
    np.testing.assert_array_equal(gm.policies[1], expected_second)
    np.testing.assert_array_equal(gm.policies[-1], expected_last)


def test_generative_model_default_e_matrix():
    """Test that the default E matrix is uniform."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]  # 2 actions
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    # T=2 with 2 actions = 4 policies
    gm = GenerativeModel(A=A, B=B, C=C, D=D, T=2)

    assert gm.num_policies == 4
    np.testing.assert_array_equal(gm.E, np.array([0.25, 0.25, 0.25, 0.25]))
    assert gm.E.dtype == np.float64

def test_generative_model_custom_e_matrix():
    """Test that a custom E matrix is preserved and converted to float64."""
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]  # 2 actions
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    # 2 policies
    E = [0.8, 0.2]

    gm = GenerativeModel(A=A, B=B, C=C, D=D, E=E, T=1)

    np.testing.assert_array_equal(gm.E, np.array([0.8, 0.2]))
    assert gm.E.dtype == np.float64


def test_generative_model_normalization():
    """Test that matrices A, B, and D are normalized appropriately."""
    # A should normalize along axis=0
    A_unnorm = [np.array([[8.0, 1.0], [2.0, 9.0]])]  # col sums: 10, 10

    # B should normalize along axis=0
    B_unnorm = [np.array([
        [[9.0, 1.0], [1.0, 9.0]],
        [[1.0, 9.0], [9.0, 1.0]]
    ]).transpose(1, 2, 0)]  # 2 actions

    C = [np.array([1.0, -1.0])]

    # D should normalize overall (or along its single axis)
    D_unnorm = [np.array([5.0, 5.0])]

    gm = GenerativeModel(A=A_unnorm, B=B_unnorm, C=C, D=D_unnorm)

    # Check A
    np.testing.assert_allclose(gm.A[0], np.array([[0.8, 0.1], [0.2, 0.9]]))

    # Check D
    np.testing.assert_allclose(gm.D[0], np.array([0.5, 0.5]))

    # Check B (columns should sum to 1)
    for a in range(2):
        for s in range(2):
            assert np.isclose(np.sum(gm.B[0][:, s, a]), 1.0)

def test_generative_model_validation():
    """Test that matrices with negative probabilities raise ValueError."""
    # Base valid matrices
    A = [np.array([[0.8, 0.2], [0.2, 0.8]])]
    B = [np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]  # 2 actions
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]

    # Invalid A
    A_invalid = [np.array([[-0.1, 1.1], [0.5, 0.5]])]
    with pytest.raises(ValueError, match="Matrix A\\[0\\] contains negative probabilities."):
        GenerativeModel(A=A_invalid, B=B, C=C, D=D)

    # Invalid B
    B_invalid = [np.array([
        [[-0.1, 1.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]]
    ]).transpose(1, 2, 0)]
    with pytest.raises(ValueError, match="Matrix B\\[0\\] contains negative probabilities."):
        GenerativeModel(A=A, B=B_invalid, C=C, D=D)

    # Invalid D
    D_invalid = [np.array([-0.5, 1.5])]
    with pytest.raises(ValueError, match="Matrix D\\[0\\] contains negative probabilities."):
        GenerativeModel(A=A, B=B, C=C, D=D_invalid)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
