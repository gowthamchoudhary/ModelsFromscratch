# LiDAR Semantic Sense From Scratch

This project is a from-scratch learning project for understanding LiDAR semantic sensing. LiDAR semantic sensing focuses on assigning a meaningful class label to each point or region in a point cloud, such as road, vehicle, pedestrian, building, vegetation, or background.

This folder currently contains the README for the project explanation. The implementation file or notebook can be added here as the project grows.

## Project Overview

LiDAR sensors produce 3D point clouds. Each point usually contains spatial information such as:

- `x`
- `y`
- `z`
- `intensity`

The goal of semantic sensing is to understand what each point represents in the real world. Instead of only detecting that an object exists, semantic sensing tries to classify the scene into useful categories.

## What the Project Covers

- Understanding LiDAR point cloud data
- Representing each point with numeric features
- Preparing labeled point cloud samples
- Extracting simple geometric features
- Classifying points or regions into semantic classes
- Comparing predicted labels with true labels
- Evaluating performance with accuracy or class-wise metrics
- Visualizing semantic labels using different colors

## Maths Behind the Idea

A LiDAR point can be represented as a feature vector:

```text
point = [x, y, z, intensity]
```

For semantic classification, the model learns a mapping:

```text
features -> semantic class
```

Example semantic classes can include:

```text
road, vehicle, pedestrian, building, vegetation
```

If using a simple distance-based classifier, nearby points can be compared using Euclidean distance:

```text
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
```

If using a probability-based classifier, the model can estimate the most likely class:

```text
predicted_class = class with highest P(class | features)
```

## How to Run

After adding an implementation file or notebook:

1. Open the `LiDAR_Semantic_Sense` folder.
2. Install the required libraries.
3. Run the Python script or notebook.
4. Check the predicted semantic labels and visualizations.

## Requirements

Common libraries for a simple LiDAR semantic sensing project can include:

```bash
pip install numpy pandas matplotlib scikit-learn
```

For point cloud visualization, you may also use:

```bash
pip install open3d
```

## What I Learned

This project helps explain the basics of LiDAR semantic understanding:

- How 3D point cloud data is represented
- Why semantic labels are useful for scene understanding
- How geometric features can help classify points
- How distance and probability can support classification
- How visualization makes point cloud predictions easier to interpret
