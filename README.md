# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!


## Program:
```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread("spider.png")

print("Face image shape:", faceImage.shape)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Face")
plt.axis("off")
plt.show()
glassPNG = cv2.imread("glass.jpg", cv2.IMREAD_UNCHANGED)

print("Original sunglasses shape:", glassPNG.shape)
if glassPNG.shape[2] == 4:

    # PNG has an alpha channel
    glassBGR = glassPNG[:, :, :3]
    glassMask1 = glassPNG[:, :, 3]

else:

    # PNG has no alpha channel
    print("WARNING: glass.jpg does not have an alpha channel.")

    glassBGR = glassPNG

    # Create a mask from non-black pixels
    gray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
    _, glassMask1 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)


# Display sunglasses and mask

plt.figure(figsize=(15, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses")
plt.axis("off")

plt.subplot(122)
plt.imshow(glassMask1, cmap="gray")
plt.title("Sunglasses Mask")
plt.axis("off")

plt.show()
# Find pixels belonging to the sunglasses
ys, xs = np.where(glassMask1 > 0)

if len(xs) == 0:
    raise ValueError("The sunglasses mask is empty.")

# Bounding box
x_min = xs.min()
x_max = xs.max()
y_min = ys.min()
y_max = ys.max()

print("Sunglasses bounding box:")
print("x:", x_min, "to", x_max)
print("y:", y_min, "to", y_max)


# Crop both image and mask
glassBGR = glassBGR[y_min:y_max+1, x_min:x_max+1]
glassMask1 = glassMask1[y_min:y_max+1, x_min:x_max+1]

print("Cropped sunglasses:", glassBGR.shape)
faceWithGlassesNaive = faceImage.copy()

x1 = 120
y1 = 120

# Width of the sunglasses on the face
new_width = 450


original_height, original_width = glassBGR.shape[:2]

new_height = int(
    original_height * new_width / original_width
)

glassBGR_resized = cv2.resize(
    glassBGR,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

glassMask1_resized = cv2.resize(
    glassMask1,
    (new_width, new_height),
    interpolation=cv2.INTER_NEAREST
)

print("Resized sunglasses:", glassBGR_resized.shape)

face_height, face_width = faceImage.shape[:2]

if x1 + new_width > face_width:
    raise ValueError("Sunglasses extend beyond the right side of the face image.")

if y1 + new_height > face_height:
    raise ValueError("Sunglasses extend beyond the bottom of the face image.")

eyeROI = faceWithGlassesNaive[
    y1:y1+new_height,
    x1:x1+new_width
]

print("Face ROI:", eyeROI.shape)
print("Glass:", glassBGR_resized.shape)

glassMask = cv2.merge(
    (
        glassMask1_resized,
        glassMask1_resized,
        glassMask1_resized
    )
)

glassMask = glassMask.astype(np.float32) / 255.0


maskedEye = cv2.multiply(
    eyeROI.astype(np.float32),
    1 - glassMask
)



maskedGlass = cv2.multiply(
    glassBGR_resized.astype(np.float32),
    glassMask
)

eyeRoiFinal = cv2.add(
    maskedEye,
    maskedGlass
)

eyeRoiFinal = np.uint8(eyeRoiFinal)


faceWithGlassesNaive[
    y1:y1+new_height,
    x1:x1+new_width
] = eyeRoiFinal

plt.figure(figsize=(20, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(np.uint8(maskedEye), cv2.COLOR_BGR2RGB))
plt.title("Masked Face")
plt.axis("off")

plt.subplot(132)
plt.imshow(cv2.cvtColor(np.uint8(maskedGlass), cv2.COLOR_BGR2RGB))
plt.title("Masked Sunglasses")
plt.axis("off")

plt.subplot(133)
plt.imshow(cv2.cvtColor(eyeRoiFinal, cv2.COLOR_BGR2RGB))
plt.title("Combined ROI")
plt.axis("off")

plt.show()

plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(122)
plt.imshow(cv2.cvtColor(faceWithGlassesNaive, cv2.COLOR_BGR2RGB))
plt.title("With Sunglasses")
plt.axis("off")

plt.show()


```


## Output:

<img width="700" height="800" alt="image" src="https://github.com/user-attachments/assets/2ba79d46-a3e4-4629-9117-da873a84dca7" />


## Result:

Thus, We successfully added Sunglasses to Your Passport Photo Using OpenCV.
