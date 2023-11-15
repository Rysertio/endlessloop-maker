import numpy as np
import cv2
from pydensecrf import densecrf

def find_1d_repetitions(image, mask, main_direction, e=8):
    # Convert the mask to binary
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Extract the region of interest (ROI) from the image using the mask
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Get the center of mass of the mask
    center_of_mass = np.array(cv2.findNonZero(mask)).mean(axis=0)[0]

    # Define the line segment along the main direction through the center of mass
    line_segment = np.linspace(center_of_mass - 50 * main_direction, center_of_mass + 50 * main_direction, 100)

    # Sample a wide band of pixels around the line segment
    sample = np.array([roi[int(round(p[1])), int(round(p[0]))] for p in line_segment])

    # Dynamic programming to find repetitions
    similarity_matrix = np.zeros((len(sample), len(sample)))

    for i in range(len(sample)):
        for j in range(len(sample)):
            if abs(i - j) > e:
                similarity_matrix[i, j] = np.linalg.norm(sample[i] - sample[j])
            else:
                similarity_matrix[i, j] = np.inf

    # Dynamic programming to find the optimal path
    path = np.zeros_like(similarity_matrix, dtype=int)

    for i in range(1, len(sample)):
        for j in range(1, len(sample)):
            if abs(i - j) > e:
                candidates = [similarity_matrix[i - 1, j - 1], similarity_matrix[i - 1, j], similarity_matrix[i, j - 1]]
                path[i, j] = np.argmin(candidates)

    # Backtrack to find the optimal path
    i, j = len(sample) - 1, len(sample) - 1
    optimal_path = []

    while i > 0 and j > 0:
        optimal_path.append((i, j))
        if path[i, j] == 0:
            i -= 1
            j -= 1
        elif path[i, j] == 1:
            i -= 1
        else:
            j -= 1

    return optimal_path

def displacement_assignment(image, mask, main_direction, optimal_path, initial_displacement):
    # Define constants
    theta_alpha = 10
    theta_beta = 1e-5
    lambda_factor = 0.1

    # Define unary data term
    def unary_data_term(pixel, displacement, initial_displacement):
        feature_vector = pixel.flatten()
        perceptual_distance = np.linalg.norm(feature_vector - (feature_vector + displacement))
        deviation_penalty = lambda_factor * (np.linalg.norm(displacement) - np.linalg.norm(initial_displacement))**2
        return perceptual_distance + deviation_penalty

    # Define pair-wise potential
    def pairwise_potential(pixel_i, pixel_j, displacement_i, displacement_j, mask_i, mask_j):
        diff_pixel = np.linalg.norm(pixel_i - pixel_j)
        diff_mask = np.linalg.norm(mask_i - mask_j)

        bilateral_weight = np.exp(-(diff_pixel**2) / (2 * theta_alpha**2) - (diff_mask**2) / (2 * theta_beta**2))

        label_similarity = np.sqrt(np.dot(displacement_i, displacement_j) / (np.linalg.norm(displacement_i) * np.linalg.norm(displacement_j)))

        return bilateral_weight * label_similarity

    # Create dense CRF
    d = densecrf.DenseCRF2D(image.shape[1], image.shape[0], len(optimal_path))
    d.setUnaryEnergy(unary_data_term)

    # Add pairwise potentials
    for i in range(len(optimal_path)):
        for j in range(len(optimal_path)):
            displacement_i = optimal_path[i]
            displacement_j = optimal_path[j]

            mask_i = mask.flatten()[i]
            mask_j = mask.flatten()[j]

            d.addPairwiseEnergy(pairwise_potential, i, j)

    # Inference to get the optimal labeling
    Q = d.inference(5)

    # Get the result as a numpy array
    result = np.array(Q).reshape((image.shape[0], image.shape[1], -1))

    return result

def warp_image(image, displacement_field, mask):
    h, w = image.shape[:2]

    # Create a new texture for warping, extending repeating pattern beyond the edges of the mask
    texture = np.zeros_like(image)
    texture[mask] = image[mask]

    # Inpainting to extrapolate the repeating pattern beyond the mask
    inpaint_mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    texture = cv2.inpaint(texture, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # Forward warp
    warped_image_forward = cv2.remap(texture, displacement_field[:, :, 0], displacement_field[:, :, 1], cv2.INTER_LINEAR)

    # Backward warp
    inverted_displacement_field = -displacement_field
    warped_image_backward = cv2.remap(texture, inverted_displacement_field[:, :, 0], inverted_displacement_field[:, :, 1], cv2.INTER_LINEAR)

    return warped_image_forward, warped_image_backward

def generate_seamless_animation(image, displacement_field_forward, displacement_field_backward, mask, alpha):
    h, w = image.shape[:2]

    animation_frames = []

    for t in np.linspace(-1, 1, num=50):  # Adjust the number of frames as needed
        if t >= 0:
            warped_image, _ = warp_image(image, t * displacement_field_forward, mask)
        else:
            _, warped_image = warp_image(image, -t * displacement_field_backward, mask)

        animation_frames.append(warped_image)

    seamless_animation = []

    for i in range(len(animation_frames)):
        t = -1 + i * 2 / (len(animation_frames) - 1)
        alpha_t = alpha(t)
        seamless_frame = alpha_t * animation_frames[i] + (1 - alpha_t) * animation_frames[(i + 1) % len(animation_frames)]
        seamless_animation.append(seamless_frame)

    return seamless_animation

def load_displacement_field(filename):
    try:
        displacement_field = np.load(filename)
        return displacement_field
    except FileNotFoundError:
        print(f"Error: Displacement field file '{filename}' not found.")
        return None

def save_displacement_field(displacement_field, filename):
    try:
        np.save(filename, displacement_field)
        print(f"Displacement field saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving displacement field to '{filename}': {e}")
