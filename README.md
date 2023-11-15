# Endless Loops: Detecting and Animating Periodic Patterns

This repository contains the implementation of the paper titled "Endless Loops: Detecting and Animating Periodic Patterns in Still Image." The paper introduces a method for detecting and animating periodic patterns in images, creating seamless looping animations.

## Overview

The implemented solution consists of several stages:

1. **1D Repetitions Detection:** Detects image repetitions along a 1D curve centered in the mask and aligned with the specified direction of motion.

2. **Displacement Assignment with CRF:** Utilizes Conditional Random Field (CRF) optimization to assign displacement vectors to pixels inside the mask, considering a set of labels obtained from the 1D repetitions.

3. **Post-processing the Displacement Field:** Improves the smoothness of the displacement field obtained from the CRF optimization using Gaussian kernel smoothing and spline interpolation.

4. **Generation of Video Frames:** Generates animation frames for a sequence of time stamps, blending forward and backward warps to create a seamless looping animation.

5. **Save and Load Displacement Fields:** Functions for saving and loading displacement fields, facilitating the reuse of precomputed fields.

## Dependencies

- Python 3.x
- NumPy
- OpenCV
- PyDenseCRF (for CRF optimization)
- Other dependencies as specified in the paper or code comments

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/endless-loops.git
   cd endless-loops
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Implementation:**
   ```bash
   python infer.py
   ```

## Notes

- Replace image filenames, mask filenames, and other parameters with your actual data.
- Adjust parameters, such as the number of frames in the animation or the alpha function, according to your preferences.

## Citation

If you use this implementation in your research or project, please consider citing the original paper:

```bibtex
@article{halperin2021endless,
  author = {Halperin, Tavi and Hakim, Hanit and Vantzos, Orestis and Hochman, Gershon and Benaim,
      Netai and Sassy, Lior and Kupchik, Michael and Bibi, Ofir and Fried, Ohad},
  title = {Endless Loops: Detecting and Animating Periodic Patterns in Still Images},
  journal = {ACM Trans. Graph.},
  issue_date = {August 2021},
  volume = {40},
  number = {4},
  month = aug,
  year = {2021},
  articleno = {142},
  numpages = {12},
  url = {https://doi.org/10.1145/3450626.3459935},
  doi = {10.1145/3450626.3459935},
  publisher = {ACM},
}
```

## License

This project is licensed under the AGPLv3 or any later version License - see the [LICENSE](LICENSE) file for details.
