import numpy as np

def embed(signal, dim, delay):
    signal_length = len(signal)
    if signal_length < (dim - 1) * delay + 1:
        raise ValueError("Signal too short for given embedding dimension and delay")
    trajectory = np.zeros((signal_length - (dim - 1) * delay, dim))
    for i in range(dim):
        trajectory[:, i] = signal[i * delay: i * delay + trajectory.shape[0]]
    return trajectory

# Step 2: Generate Recurrence Plot
def recurrence_plot(trajectory, threshold):
    L, m = trajectory.shape
    R = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            # Euclidean distance between points in the phase space (2-norm)
            distance = np.linalg.norm(trajectory[i] - trajectory[j])
            # Check if the distance is below the threshold
            R[i, j] = 1 if distance <= threshold else 0

    return R.astype(int)

# Define a function to calculate determinism
def calculate_determinism(rp_matrix, min_diagonal_length=2):
    # Ensure that diagonal elements are not considered
    np.fill_diagonal(rp_matrix, 0)

    # Prepare a shifted matrix accumulator to count diagonal lines
    diagonals_count = np.zeros(rp_matrix.shape[1], dtype=int)

    # Iterate over possible diagonal shifts
    for shift in range(1, rp_matrix.shape[1]):
        # Create shifted versions of the matrix
        shifted_diagonal = np.diagonal(rp_matrix, offset=shift)

        # Start and end flags for consecutive ones
        start_flag = False

        # Diagonal line length counter
        line_length = 0

        # Count consecutive ones which correspond to diagonal lines in the RP
        for i in range(shifted_diagonal.size):
            if shifted_diagonal[i] == 1 and not start_flag:
                # Start of a new line
                start_flag = True
                line_length = 1
            elif shifted_diagonal[i] == 1 and start_flag:
                # Continuation of the current line
                line_length += 1
            elif shifted_diagonal[i] == 0 and start_flag:
                # End of the current line
                start_flag = False
                if line_length >= min_diagonal_length:
                    diagonals_count[line_length] += 1
                line_length = 0
        # Handle case where line continues until the last element
        if line_length >= min_diagonal_length:
            diagonals_count[line_length] += 1

    # Calculate the determinism (number of points in diagonal lines of at least min_diagonal_length / total number of recurrences)
    determinism = sum(diagonals_count[min_diagonal_length:] * np.arange(min_diagonal_length, len(diagonals_count))) / np.sum(rp_matrix)

    return determinism

def CalcDET(signal, embed_dim, delay_time, threshold, min_l):
    embeded = embed(signal, embed_dim, delay_time)
    R = recurrence_plot(embeded, threshold)
    determinism = calculate_determinism(R, min_l)
    return determinism