### README for Running Tests

---

#### Overview

Instructions for running tests for the `dsa110-nsfrb` repository:
1. `imaging.py`: Handles imaging functions.
2. `searching.py`: Contains helpers for dedispersion and matched filter searching.
3. `classifying.py`: Deals with the classification of images.

The tests are organized in the `tests` directory and can be executed using `pytest`. Ensure `pytest` is installed and execute the tests inside the `tests` directory.

---

#### Running Tests

To run the tests, navigate to the `tests` directory and execute the following commands:

1. **Run Imaging Tests**

   ```sh
   pytest test_imaging.py
   ```

   This will execute the tests for the `imaging.py` module. You should see output indicating that all tests have passed.

2. **Run Searching Tests**

   ```sh
   python test_searching.py
   ```

   This will execute the tests for the `searching.py` module. You should see output indicating that all tests have passed.

3. **Run Classifying Tests**

   ```sh
   pytest test_classifying.py
   ```

   This will execute the tests for the `classifying.py` module. You should see output indicating that all tests have passed.

---

#### Details of Tests



1. Imaging Tests (`test_imaging.py`):

- **Test Briggs Weighting**: Ensures that the `briggs_weighting` function returns valid weights.
- **Test Robust Image**: Checks the output of the robust imaging function with complex visibility data.
- **Test Uniform Image with Delta Visibility**: Ensures that a delta function in the visibility domain produces an approximately uniform image.
- **Test Uniform Image with Constant Visibility**: Ensures that constant visibility produces an image with a peak at the center.

2. Searching Tests (`test_searching.py`):
- **Test Baseline Search**: Runs search with baseline parameters (no FFTs, no multithreading)
- **Test FFT Search**: Runs search with FFT spatial matched filtering
- **Test Multithreading Search**: Runs search with FFT and multithreading using concurrent.futures module

3. Classifying Tests (`test_classifying.py`):

- **Test Numpy Image Cube Dataset**: Verifies the custom dataset for loading image cubes.
- **Test Enhanced CNN**: Ensures the Enhanced CNN model produces the correct output shape.
- **Test Classify Images**: Checks the classification function with dummy data.
- **Test Classify Known Images**: Verifies the classification accuracy and probability checks for known data.
---
