# Test images

Five images from the **held-out test split**, with their labels in
`ground_truth.json`.

They were chosen by filtering on the `test` list in the damage run's
`splits.json`, which is the same split the evaluation used, so none of them was
trained on. Within that split they are the richest: 9 to 22 labelled instances and
5 to 6 distinct damage types each, because one faint scratch is a poor test.

Source: the Kaggle car-damage dataset used to train the detectors. Included here
because the brief asks for the test images and their outputs; they are not the
project's own work.

To regenerate the outputs:

    python -m scripts.run_test_images

Needs `pip install -e ".[vision]"` and the two weights files in `weights/`.
