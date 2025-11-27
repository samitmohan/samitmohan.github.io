YOLO


---
layout: post
title: "Understanding Object Detection"
date: 2025-11-21
---

Say you're on the street, in just miliseconds you spot a car, man, traffic light, dogs etc.. you don't just see a mash of pixels, you see *objects* 
You know where they are, what they are and how they're moving. How can you teach a computer the same thing?

For a long time, teaching a computer to do this was a painfully slow process. Early models were like a detective with a magnifying glass, scanning an image patch by patch, asking, "Is this a car? No. Is *this* a car? No." It worked, but it was slow. Painfully slow. You could brew a pot of coffee in the time it took to analyze one frame of video.

Then, in 2016, a paper came out with a name that was as bold as its idea: **You Only Look Once (YOLO)**.

The name says it all. What if, instead of looking at an image thousands of times, a computer could just... look once? This post is the story of that idea. We'll break down how YOLOv1 works, piece by piece, using simple terms. We'll build it up from scratch, figure out *why* it works, and see where it stumbles. Then, we'll take a quick trip through time to see how this revolutionary idea evolved into the state-of-the-art YOLOv10.

## Table of Contents
0. [Pipeline](#pipeline)
1. [Before YOLO](#before-yolo)
2. [Why YOLO: The Motivation](#why-yolo-the-motivation)
3. [YOLO High Level Overview](#yolo-high-level-overview)
4. [YOLO Architecture](#yolo-architecture)
5. [Grid Cells & Predictions](#grid-cells-and-predictions)
6. [Bounding Box Format & Encoding](#bounding-box-format-and-encoding)
7. [Prediction Vector Breakdown](#prediction-vector-breakdown)
8. [Training Process](#training-process)
9. [Loss Function Explained](#loss-function-explained)
10. [Inference: From Predictions to Bounding Boxes](#inference-from-predictions-to-bounding-boxes)
11. [Post-Processing: IOU & NMS](#post-processing-iou-and-nms)
12. [Evaluation Metrics: mAP](#evaluation-metrics-map)
13. [Performance & Results](#performance-and-results)
14. [Limitations of YOLOv1](#limitations-of-yolov1)
15. [Evolution: From v1 to v10](#evolution-from-v1-to-v10)
16. [Implementation: PyTorch Code](#implementation-pytorch-code-examples)
17. [Extra](#extra)

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOLO OBJECT DETECTION                            │
└─────────────────────────────────────────────────────────────────────────┘

Input Image (448×448)
      │
      ├──► Step 1: Grid Division
      │          ┌───┬───┬───┬───┬───┬───┬───┐
      │          │   │   │   │   │   │   │   │     7×7 grid
      │          ├───┼───┼───┼───┼───┼───┼───┤     Each cell = 64×64 pixels
      │          │   │ P │   │   │   │   │   │     P = Person center
      │          ├───┼───┼───┼───┼───┼───┼───┤     D = Dog center
      │          │   │   │   │ D │   │   │   │
      │          └───┴───┴───┴───┴───┴───┴───┘
      │
      ├──► Step 2: CNN Forward Pass (24 conv layers)
      │          Features extracted: 7×7×1024
      │
      ├──► Step 3: Predictions per Cell
      │          Each cell outputs 30 values:
      │          • 2 boxes: (x,y,w,h,conf) × 2 = 10 values
      │          • 20 class probabilities = 20 values
      │          Total: 7×7×30 = 1,470 predictions
      │
      ├──► Step 4: Post-Processing
      │          • Filter by confidence (threshold = 0.2)
      │          • Apply Non-Maximum Suppression (NMS)
      │          • Keep top predictions per class
      │
      └──► Final Output
               ┌──────────────────┐
               │  Person: 95.3%   │  [Bounding boxes with labels]
               │  Dog: 87.6%      │
               └──────────────────┘
```

### Metrics

| Metric | Value | Comparison |
|--------|-------|------------|
| **Speed** | 45 fps | 9× faster than Faster R-CNN |
| **Accuracy** | 63.4% mAP | -7% vs Faster R-CNN |
| **Grid Size** | 7×7 cells | 49 possible detections |
| **Parameters** | ~20M | ResNet34 backbone |
| **Input Size** | 448×448 | Fixed resolution |

---

## The "Old" Way: A Tale of Two Stages

Before YOLO burst onto the scene, the world of object detection was dominated by a family of models called R-CNN (Regions with CNN features). These were clever, but oh-so-methodical.

Think of it like an overly cautious security guard trying to identify people in a photograph.

**Stage 1: Propose Regions (The "Hey, something's here!" stage)**

First, the guard doesn't look at the whole picture. Instead, they draw about 2,000 boxes all over it, around anything that looks remotely interesting. "This might be a person. This shadow could be a person. This oddly shaped bush is probably not a person, but let's draw a box around it just in case." These boxes are called "region proposals."

**Stage 2: Classify and Refine (The "Let me take a closer look" stage)**

Next, the guard takes every single one of those 2,000 boxes and feeds each one into a powerful (and slow) classifier. For each box, it asks:

1.  **What is this?** (Is it a person, a car, or that bush from before?)
2.  **Can I make this box fit better?** (The initial box was a bit sloppy, let's tighten it up.)

This two-stage process, especially its faster successor "Faster R-CNN," was accurate. Very accurate. But it was also a pipeline of separate, complex steps.

**The Big Problem:** It was SLOOOOW. At around 7 frames per second, it was nowhere near fast enough for real-time video. It was like watching a movie one slide at a time.

### Comparison: Object Detection Methods (2014-2016)

| Method | Year | Speed (fps) | mAP (%) | Pipeline Type | Proposals | Real-time? |
|--------|------|-------------|---------|---------------|-----------|------------|
| **R-CNN** | 2014 | 0.02 | 66.0 | Two-stage | Selective Search (2000+) | ❌ |
| **Fast R-CNN** | 2015 | 0.5 | 70.0 | Two-stage | Selective Search | ❌ |
| **Faster R-CNN** | 2015 | 7 | 73.2 | Two-stage | RPN (300) | ❌ |
| **DPM** | 2015 | <1 | 30.4 | Sliding window | Dense sampling | ❌ |
| **YOLOv1** | 2016 | **45** | 63.4 | **Single-stage** | **None** | **✅** |

---

## The YOLO Breakthrough: Just Look Once

This is where YOLO's creators asked a brilliant question: "What if we stop this two-step nonsense? What if we could just look at the image once and get the answers?"

It was a radical idea. Instead of a careful, multi-stage pipeline, they proposed a single, elegant network that does everything at once.

It treats object detection as a single regression problem. Don't worry about the term "regression"—all it means is that instead of a multi-step process, the network just spits out a bunch of numbers that directly describe the bounding boxes and class probabilities.

**The Old Way (R-CNN):**
1.  Generate ~2000 region proposals.
2.  For each proposal, run a classifier.
3.  Refine the boxes.

**The YOLO Way:**
1.  Look at the image. Once.
2.  Done.

This approach had huge advantages:

*   **It's FAST.** Like, *really* fast. We're talking 45 frames per second fast. Real-time video was suddenly on the table.
*   **It sees the whole picture.** Unlike the R-CNN guard who only looked at tiny patches, YOLO sees the entire image at once. This gives it context, making it less likely to mistake a patch of background for an object.
*   **It's simpler.** One network, one training process. End-to-end.

Of course, there was a trade-off. In the beginning, YOLO was a little less accurate than its two-stage cousins. It was like a rookie guard who's incredibly fast but sometimes misses a detail or two. But the speed it unlocked changed the game forever.

*(A quick note: The original YOLO paper was by Joseph Redmon et al. Ultralytics is the company behind the popular YOLOv5, v8, and now v10 models, and they've done amazing work to make YOLO accessible and powerful.)*

---

## How YOLO Works: A Game of Battleship

So how does it "look once"? The core idea is surprisingly simple and elegant. It's like a high-stakes, super-fast game of Battleship.

Here's the game plan:

1.  **Take the image and lay a grid over it.** The original YOLO uses a 7x7 grid. This divides the image into 49 cells.

2.  **Each grid cell gets a job.** It's responsible for detecting any object whose *center point* falls within that cell. It's like each square on the Battleship board is responsible for reporting if the center of a ship is in it.

3.  **Make a prediction for every cell.** A single, powerful neural network looks at the whole image and, for every single one of the 49 grid cells, it spits out a prediction. This prediction answers a few key questions:
    *   "Is an object's center in here? How confident am I?"
    *   "If so, where is the bounding box for that object?"
    *   "And what class is it (a dog, a person, a car)?"

That's it. One look, one pass through the network, and out comes a flood of predictions from all 49 cells at once. No proposals, no second stage. Just a direct mapping from image pixels to bounding boxes and class probabilities.

![YOLO High-Level Pipeline](/assets/images/yolo/basic.png)
*Figure: YOLO's single-stage detection pipeline - the entire image is processed once to produce bounding boxes and class predictions*

![YOLO Algorithm Overview](/assets/images/yolo/yoloAlgo.png)
*Figure: The complete YOLO detection algorithm from input image to final predictions*

---

## The "Brain" of YOLO: The Architecture

At its heart, YOLO is a Convolutional Neural Network (CNN). If you're new to CNNs, don't sweat it. Think of it like a company's hierarchy, designed to process information from the ground up.

1.  **The Interns (Early Layers):** The first few layers of the network are like interns. They get the raw image and have a very simple job: find basic patterns. One intern looks for vertical edges, another for horizontal edges, another for patches of green, and so on. They're not smart, but they're fast and there are a lot of them.

2.  **The Mid-Level Managers (Deeper Layers):** These layers take the simple reports from the interns and combine them into more complex ideas. A manager might see reports for a sharp corner, a curved line, and a circular shape and say, "Hmm, that's starting to look like an eye." Another might combine reports of four straight lines and a rectangle to identify a "window."

3.  **The CEO (Final Layers):** The final layers are the executives. They take the high-level reports from the managers ("we've got an eye," "we've got a furry texture," "we've got a wet nose") and make the final call: "That's a dog."

YOLO's architecture is essentially a 24-layer version of this, followed by 2 "fully connected" layers which act like the final board meeting where all the information is synthesized into the final output.

![YOLO Architecture](/assets/images/yolo/arch.png)
*Figure: The YOLOv1 network architecture. Think of it as information flowing from left (interns) to right (CEO).*

A few clever tricks make this "company" efficient:

*   **Leaky ReLU:** This is an office rule. A normal ReLU activation is like an employee who stays completely silent if they have nothing positive to report. This can lead to "dying neurons" (employees who never speak up again). Leaky ReLU is a rule that says, "Even if your input is negative, report a tiny fraction of it." This keeps the information flowing and prevents parts of the network from dying.
*   **1x1 Convolutions:** These are like hyper-efficient department heads. Before passing information up to the next level, they summarize and condense it, reducing the amount of work the next layer has to do. It's a clever way to add complexity without adding a ton of computational cost.
*   **Dropout:** This is like randomly telling some employees to take the day off during training. It sounds crazy, but it forces the other employees to learn to cover for them, making the whole system more robust and less reliant on any single "star employee" (neuron). This prevents overfitting.

Ultimately, this whole structure is designed to take a 448x448 pixel image and transform it into a meaningful 7x7x30 tensor that holds all our predictions. We'll decode that tensor next.

---

## Back to Battleship: Grid Cells and Responsibility

Remember our 7x7 grid, like a Battleship board laid over the image? Let's get specific about the rules of the game.

**The Golden Rule: The Center of the Object Decides.**

Each of the 49 grid cells has one job: it is **responsible** for detecting an object *if the center of that object falls inside it*.

![Grid Cells and Center Points](/assets/images/yolo/grid_cell_and_center.png)
*Figure: The image is divided into a 7x7 grid. The cell containing the dog's center point (blue dot) is responsible for detecting the dog. The cell with the person's center is responsible for the person.*

If a person's center is in cell (3, 4), that cell is in charge of predicting the person. If a car's center is in cell (6, 1), that cell has to predict the car. The other 47 cells can relax (sort of).

**But there's a catch... a big one.**

What happens if the center of *two* objects falls into the same grid cell? Imagine a person standing directly in front of a car.

**YOLOv1's limitation: Each grid cell can only detect ONE object.**

This is a fundamental rule of the original YOLO. Even though each cell will propose two bounding boxes (we'll get to that), it can only have one final opinion on the class of the object. It's like a square on the Battleship board that can only shout "Hit!" for one ship, even if two ships happen to cross over it.

This is YOLOv1's biggest weakness. It struggles with crowds or flocks of birds where multiple small objects are clustered together. This is a key problem that later versions of YOLO were desperate to solve. For now, just remember: one cell, one final vote.

---

## Giving Directions: How YOLO Describes a Bounding Box

Okay, so a grid cell is responsible for an object. But how does it describe *where* the object is?

A naive approach would be to predict the absolute pixel coordinates, like `(x=100, y=200, width=130, height=202)`. But these numbers are big and can vary wildly. A neural network would have a tough time learning to predict them directly from scratch. It's like trying to guess someone's exact GPS coordinates.

YOLO does something much smarter. It gives directions, relative to its own little neighborhood (the grid cell).

Let's say the cell at `(row=3, col=4)` is responsible for detecting a dog. Here's how it describes the dog's bounding box:

1.  **Center Point (`x`, `y`): Where is the center *inside this cell*?**
    Instead of global coordinates, it predicts two numbers between 0 and 1.
    *   `x = 0.5` would mean the dog's center is horizontally in the middle of this cell.
    *   `y = 0.2` would mean the dog's center is 20% of the way down from the top of this cell.
    It's like saying, "From my top-left corner, go over 50% and down 20%." These small, normalized values are much easier for a network to predict.

2.  **Dimensions (`w`, `h`): How big is the box *relative to the whole image*?**
    For the width and height, YOLO uses a different reference. It predicts the box's size as a fraction of the entire image.
    *   `w = 0.3` means the dog's bounding box is 30% of the total image width.
    *   `h = 0.2` means the box is 20% of the total image height.

So, the final "directions" are four simple, normalized numbers: `[x_offset, y_offset, width_fraction, height_fraction]`.

![Center Calculations](/assets/images/yolo/calculations_on_cneter.png)
*Figure: The key idea: the `x` and `y` coordinates are offsets relative to the grid cell (the small square), while `w` and `h` are relative to the entire image.*

This encoding is the secret sauce. It transforms a difficult "guess the exact coordinates" problem into a much more manageable "predict a few small numbers" problem. This makes training the network stable and effective.

## The Full Report: Decoding the 7x7x30 Tensor

We know the network spits out a `7x7x30` block of numbers. Let's zoom in on one of those 49 grid cells and look at the 30 numbers it's responsible for predicting. Think of it as the official report form each cell has to fill out.

For a model trained on the Pascal VOC dataset (which has 20 classes), that 30-number vector is broken down like this:

---

**Part 1: The First Bounding Box Guess (5 numbers)**
*   `[x, y, w, h, c]`
*   `x, y, w, h`: These are the "directions" we just talked about. Four numbers describing one potential bounding box.
*   `c`: This is the **confidence score** for this specific box. It's a crucial number that answers two questions at once:
    1.  How likely is it that there's *any* object in this box?
    2.  If there is, how good is the fit of my box?
    This is formally `Confidence = Pr(Object) * IOU(pred, truth)`. A high confidence score means "I'm pretty sure there's an object here, and my box is a great fit."

**Part 2: The Second Bounding Box Guess (5 numbers)**
*   `[x, y, w, h, c]`
*   YOLOv1 gives each grid cell *two* chances to predict a bounding box. Why? To handle objects with different shapes. One box might be tall and skinny (good for a person standing up), while the other might be short and wide (good for a car). The network learns to specialize these two "predictor" boxes.

**Part 3: The Class Prediction (20 numbers)**
*   `[p_class1, p_class2, ..., p_class20]`
*   This is a list of probabilities, one for each class (person, car, dog, etc.).
*   **Crucially, this is a *conditional* probability.** It answers the question: "**Given that there is an object in this grid cell**, what is the probability that it's a person? A car? A dog?"
*   Notice there's only ONE set of class probabilities for the whole cell, which is shared by BOTH bounding box guesses. This is the reason for the "one object per cell" limitation we talked about earlier.

---

So, to recap, for each of the 49 cells, the network predicts:
`(5 numbers for box 1) + (5 numbers for box 2) + (20 class probabilities) = 30 numbers.`

The entire output is a `7x7x30` tensor, which is a compact and powerful representation of all the potential objects in the image. Now, we just have to figure out how to interpret this massive block of numbers.

---

## From Numbers to Boxes: The Inference Process

Okay, the network has done its job and given us a giant `7x7x30` block of predictions. Right now, it's just a mess of numbers. Our job is to turn this mess into a clean list of final detections. This process is called **inference**.

Here's how we do it, step-by-step:

**Step 1: Calculate the "Final Score" for each box.**

For each of the 98 boxes (49 cells * 2 boxes per cell), we need a final score that tells us how confident we are that this box contains a specific object (like a dog or a car).

We get this by multiplying two things:
*   The box's confidence score (`c`): "How sure am I that *any* object is in this box?"
*   The highest class probability (`p`): "Assuming there is an object, how sure am I that it's a *dog*?"

`Final Score = Box Confidence (c) * Highest Class Probability (p)`

For example, if a box has a confidence of 0.8 and the highest class probability is 0.9 for "person", its final score for "person" is `0.8 * 0.9 = 0.72`.

**Step 2: Throw away the junk.**

Most of these 98 boxes will be garbage. They'll have very low final scores. So, we set a threshold (say, 0.25) and discard any box with a score below that. This is like throwing away all the blurry, nonsensical photos from a photoshoot. This dramatically reduces the number of boxes we have to deal with.

**Step 3: Convert from "YOLO-speak" to pixel coordinates.**

The remaining boxes are still in YOLO's relative "directions" format. We need to convert them back into actual pixel coordinates that we can draw on the screen. This is just a matter of reversing the math from before: we use the grid cell's position and the image dimensions to turn the relative `[x, y, w, h]` into absolute `[x1, y1, x2, y2]` pixel values.

![Converting Coordinates](/assets/images/yolo/gt_coord.png)
*Figure: We reverse the process, taking the relative predictions and the grid cell's location to calculate the final pixel coordinates of the box.*

**Step 4: Deal with duplicates (Non-Maximum Suppression).**

We've filtered out the low-confidence boxes, but we still have a problem: multiple boxes might be detecting the same object. This is where a crucial step called **Non-Maximum Suppression (NMS)** comes in. We'll give it its own section because it's that important.

---

## The Final Cleanup: IOU and Non-Maximum Suppression (NMS)

After step 3 of inference, we have a decent set of boxes, but it's still messy. Because neighboring grid cells might all be trying to detect the same object, we often end up with a cluster of overlapping boxes around a single object.

This is where a crucial step called **Non-Maximum Suppression (NMS)** comes in, but first, we need to understand its core metric.

### Intersection over Union (IOU)

**What is IOU?** It's a simple and brilliant metric that measures how much two bounding boxes overlap. It's a score from 0.0 (no overlap) to 1.0 (perfect overlap).

**Formula:**
`IOU = Area of Overlap / Area of Union`

It's the area of the intersection of the two boxes, divided by the total area they cover together.

![IOU Visualization](/assets/images/yolo/iou.png)
*Figure: IOU is the ratio of the green area (intersection) to the total colored area (union).*

We use IOU for two key things:
1.  To judge if a prediction is correct during evaluation (is its IOU with the ground truth box > 0.5?).
2.  To help us clean up duplicate detections during NMS.

### Non-Maximum Suppression (NMS): The Decluttering Step

Now, back to the mess. We have multiple boxes for the same object. How do we pick just one? We use NMS, which is like a "survival of the fittest" competition for bounding boxes.

![NMS Process](/assets/images/yolo/nms.png)
*Figure: Before NMS, you have multiple messy detections for the same object. After NMS, you have one clean, confident detection.*

Here's how it works for a specific class (e.g., for all "person" boxes):

1.  **Find the "champion" box:** Take the list of all "person" boxes and sort them by their final score, from highest to lowest. The one at the top is our champion. Keep it.

2.  **Challenge the others:** Now, go down the rest of the list. For every other box, compare it to the champion box using their IOU score.

3.  **Suppress the losers:** If any other box has a high IOU with the champion (say, > 0.5), it means it's basically detecting the *same person*. We "suppress" it—we get rid of it. It's a redundant guess.

4.  **Repeat:** Once you've gone through the list, the champion from round 1 is safe. Now, take the next-highest-scoring box that *wasn't* suppressed and declare it the champion of round 2. Repeat the process until you have no boxes left to check.

The result? You're left with only the best, most confident box for each individual object. It's a beautifully simple way to clean up the final predictions.

> **Think about it: What's the risk of setting the NMS threshold too high or too low?**
> *   If the IOU threshold is **too low** (e.g., 0.2), you might accidentally suppress boxes for two different people who are just standing close to each other.
> *   If the IOU threshold is **too high** (e.g., 0.9), you might fail to suppress a slightly-offset duplicate box, leaving you with messy results.
> The standard value of 0.5 is usually a good compromise.


---

## How YOLO Learns: The Training Process

We've seen what YOLO does, but how does it get so smart? Like any machine learning model, it has to be trained. This is the process of showing it thousands of examples and slowly teaching it to go from clueless to clever.

The YOLOv1 authors used a two-stage training process, which is like sending the network to school and then to a specialized job.

**Stage 1: "Grade School" on ImageNet**

First, they take the first 20 layers of the YOLO network (the "interns" and "managers") and train them on a massive dataset called ImageNet. The task is simple: image classification. The network is shown millions of images and just has to say "this is a cat," "this is a car," "this is a chair."

Why do this? It's like teaching a child to recognize basic shapes and objects before asking them to describe a complex scene. This pre-training teaches the network to see general visual features—edges, corners, textures, colors. It gives the model a solid foundation of what the visual world looks like.

**Stage 2: "Specialist Training" on Pascal VOC**

Once the network has its "bachelor's degree" in seeing, the full network (all 24 layers plus the final decision-making layers) is trained on the real task: object detection. They use the Pascal VOC dataset, which has images with labeled bounding boxes for 20 different object classes.

Here, the network learns to apply its visual knowledge to the much harder task of drawing boxes and classifying objects simultaneously. They also double the input image resolution to 448x448, because drawing precise boxes requires seeing finer details.

### The Secret to Good Training: Data Augmentation

If you only show a model pictures of red cars, it will be terrible at detecting blue cars. To make the model robust, you have to show it a wide variety of examples. **Data augmentation** is the process of taking your existing training images and creating new, slightly modified versions of them.

YOLO does this aggressively:
*   **Random scaling and translation:** It zooms in and out and shifts the objects around.
*   **Random color adjustments:** It messes with the exposure and saturation, so the model learns that a "dog" is still a "dog" in bright sunlight or in the shade.

This process prevents the model from just memorizing the training data and helps it generalize to new, unseen images. It's the training equivalent of "what doesn't kill you makes you stronger."

---

## The Soul of YOLO: The Loss Function

This is the most important part of the whole system. The loss function is YOLO's "teacher." During training, the network makes a prediction, and the teacher's job is to look at that prediction, compare it to the correct answers (the "ground truth"), and give the network a single score of "how wrong you were." This score is the loss. The network then uses this score to adjust its internal wiring to do better next time.

But grading this test is tricky. The teacher needs to be smart about how it assigns points. A simple "right vs. wrong" isn't enough. The YOLO loss function is a "multi-part essay question," not a simple true/false test. It's composed of three main types of error:

1.  **Localization Loss:** How wrong were your bounding box coordinates? (The "Geometry" score)
2.  **Confidence Loss:** How wrong was your "objectness" confidence score? (The "Self-Awareness" score)
3.  **Classification Loss:** Did you pick the right class for the object? (The "Multiple Choice" score)

Let's see how the teacher grades each part.

### The Big Problem: Most of the Image is Empty

Imagine a test with 49 questions, but for 47 of them, the answer is "nothing to see here." Only 2 questions are about actual objects. If the teacher grades every question equally, the student (our network) will quickly learn a lazy strategy: just write "nothing to see here" for all 49 questions. It will get 47/49 right, a score of 96%! The student will be happy, but it will have learned nothing about finding objects.

This is the **class imbalance** problem. Most grid cells in an image are background. If we're not careful, the network will learn to only predict "no object" and the loss from the few cells that *do* contain objects will be drowned out.

**The Teacher's Solution: Weighted Grading.**

The YOLO teacher is smart. It announces two new grading rules:
*   **Rule #1 (`λ_coord = 5`):** Getting the box coordinates right is *really* important. I'll give you **5x bonus points** for this part.
*   **Rule #2 (`λ_noobj = 0.5`):** For the questions where the answer is "nothing," I'll only penalize you **half a point** for getting it wrong.

This focuses the network's attention: "Pay extra attention to getting the boxes right, and don't worry so much about the empty background cells."

### Part 1: The Geometry Grade (Localization Loss)

This part grades the bounding box `(x, y, w, h)`. But again, the teacher is clever. A 10-pixel error is a huge mistake for a tiny object, but barely noticeable for a massive one. Grading them the same would be unfair.

**The Teacher's Solution: Grade the Square Root.**

Instead of grading the width `w` and height `h` directly, the teacher grades their **square roots (`√w` and `√h`)**.

Why? Think about it:
*   For a small box (width = 25), `√25 = 5`. An error of 10 pixels (new width = 35) leads to `√35 ≈ 5.9`. The error in the square-root world is `5.9 - 5 = 0.9`.
*   For a large box (width = 400), `√400 = 20`. An error of 10 pixels (new width = 410) leads to `√410 ≈ 20.25`. The error in the square-root world is `20.25 - 20 = 0.25`.

The same 10-pixel mistake creates a much bigger error for the small box! This forces the network to be much more precise with small objects, which is exactly what we want.

![Complete Loss Function](/assets/images/yolo/entire_loss.png)
*Figure: The complete loss function. It looks scary, but it's just our teacher's grading rubric written down formally.*

### Part 2: The Self-Awareness Grade (Confidence Loss)

This grades the confidence score `c`.

*   **If an object is in the cell:** The teacher wants the network's confidence to reflect how good its box is. The "correct" confidence score is the IOU between the predicted box and the ground truth box. If the network draws a perfect box (IOU=1.0), it should be 100% confident. If it draws a sloppy box (IOU=0.4), it should say it's only 40% confident.
*   **If no object is in the cell:** This is simple. The teacher wants the network to be very confident that there's nothing there. The target confidence is 0. (And remember, this mistake is only worth half a point, thanks to `λ_noobj`).

### Part 3: The Multiple Choice Grade (Classification Loss)

This is the easiest part to understand. If, and only if, a cell contains an object, the teacher looks at the 20 class probabilities. It checks if the network assigned the highest probability to the correct class. If the object was a dog, but the network said "90% cat," it's going to lose a lot of points here.

This loss is *not* calculated for cells with no objects. The teacher doesn't care what class you predict for an empty patch of sky.

### The Final Grade

The total loss, or the final grade on the test, is the sum of all these parts:

`Total Loss = (λ_coord * Geometry_Loss) + (Confidence_Loss_for_Objects + λ_noobj * Confidence_Loss_for_Background) + Classification_Loss`

This carefully designed loss function is the "soul" of YOLO. It pushes the network to not just find objects, but to find them precisely, to be aware of its own accuracy, and to classify them correctly, all while not getting distracted by the boring background.

---

## The Final Exam: How Do We Grade an Object Detector?

So, our model is trained and it's making predictions. How do we know if it's any good? We need a final exam. In object detection, the standard "exam" is called **mAP (mean Average Precision)**.

To understand mAP, we first need to understand two simpler concepts: Precision and Recall.

Imagine our model is taking a test. The test image has 10 dogs in it. The model detects 8 boxes that it claims are dogs.

*   Of those 8 boxes, 6 are actually correct (they overlap a real dog with an IOU > 0.5). These are **True Positives (TP)**.
*   The other 2 boxes were mistakes (e.g., it drew a box around a bush). These are **False Positives (FP)**.
*   The model completely missed 4 of the real dogs. These are **False Negatives (FN)**.

With these numbers, we can ask two questions:

1.  **Precision: "Of the answers you gave, how many were right?"**
    *   `Precision = TP / (TP + FP) = 6 / (6 + 2) = 75%`
    *   This measures how trustworthy the model's predictions are. A high precision model doesn't make many silly mistakes.

2.  **Recall: "Of all the things you *should* have found, how many did you find?"**
    *   `Recall = TP / (TP + FN) = 6 / (6 + 4) = 60%`
    *   This measures how comprehensive the model is. A high recall model doesn't miss much.

**The Inevitable Trade-off**

There's a constant tug-of-war between precision and recall.
*   If you're very timid and only make a prediction when you're 100% sure, you'll have high precision but low recall (you'll miss a lot).
*   If you're very bold and make tons of guesses, you'll have high recall but low precision (you'll have a lot of junk predictions).

### So, What is mAP?

A good model needs to be both precise and comprehensive. To measure this, we can't just look at one point. We need to see how precision and recall change as we adjust our confidence threshold.

**Average Precision (AP)** is a single number that summarizes this trade-off for a *single class* (like "dog"). It's the area under the precision-recall curve. A high AP means the model maintains high precision even as it tries to find more objects (i.e., as recall increases).

**Mean Average Precision (mAP)** is simply the average of the AP scores across all classes. If you have 20 classes, you calculate the AP for dogs, the AP for cats, the AP for cars, and so on, and then average them all together.

This gives us a single, comprehensive number to judge the overall performance of our model. It's the final GPA for our object detector.


---

## The Report Card: How Did YOLOv1 Do?

So, after all that, what was the verdict on YOLOv1 back in 2016? Here's the report card.

### Grade A+: Speed

This is where YOLOv1 absolutely crushed the competition.
*   **YOLOv1:** 45 frames per second (fps).
*   **Faster R-CNN (the previous champ):** ~7 fps.

This wasn't just an improvement; it was a revolution. 45 fps meant you could run object detection on a live video feed with no problem. A smaller version, Fast-YOLO, even hit a blistering 155 fps. For the first time, true real-time object detection was possible.

### Grade B-: Accuracy

With great speed came a small sacrifice in accuracy.
*   **YOLOv1:** 63.4% mAP
*   **Faster R-CNN:** ~70-73% mAP

YOLO was less accurate than the slower, more methodical two-stage detectors. It made more localization errors (the boxes weren't as perfectly placed) and struggled to find all the objects (lower recall). But for many applications, this trade-off was more than worth it.

### Special Award: Generalization

One surprising strength of YOLO was its ability to generalize. Because it sees the entire image at once, it learns a more holistic representation of objects. This meant it was less likely to be fooled by background noise. When shown artwork or unusual images, YOLOv1 made fewer silly mistakes than its competitors. It had better "common sense."

![YOLOv1 Results](/assets/images/yolo/results.png)
*Figure: The key trade-off. YOLO (top left) is way faster than competitors like Faster R-CNN, but has a slightly lower mAP.*

**The Verdict:** YOLOv1 was a game-changer. It proved that a single-stage detector could be fast enough for real-time applications while maintaining competitive accuracy. It wasn't perfect, but it laid the foundation for a decade of innovation.

---

## YOLOv1's Kryptonite: The Weak Spots

YOLOv1 was brilliant, but it wasn't perfect. Its speed came from a set of strict rules it had to follow, and these rules created some very specific weaknesses.

### 1. Crowds and Small Objects

This is YOLOv1's biggest weakness. Remember the rule: **one grid cell can only detect one object**.

*   **The Problem:** If you have a flock of birds, or a dense crowd of people, multiple objects will have their centers in the same grid cell. YOLOv1 will only pick one of them, completely missing the others. It's also just bad at detecting tiny objects in general, as they can get lost in the coarse 7x7 grid.
*   **The Fix in Later Versions:** Later YOLOs (like YOLOv3) use much finer grids and predict at multiple scales, allowing them to detect thousands of objects of all sizes.

### 2. Weirdly Shaped Objects

YOLOv1 learns to predict bounding boxes based on the shapes it sees in the training data.

*   **The Problem:** If it's mostly trained on cars with a "normal" aspect ratio, it will struggle to detect a long, skinny limousine or a tall, narrow sign. It tries to fit its standard-shaped boxes to these unusual objects and fails.
*   **The Fix in Later Versions:** YOLOv2 introduced "anchor boxes"—a set of pre-defined box shapes of different aspect ratios. This gave the network a better starting point for predicting objects of various shapes.

### 3. Imprecise Boxes (Localization Errors)

*   **The Problem:** Compared to the slow and steady two-stage detectors, YOLO's boxes were often a bit... sloppy. The final prediction comes from a coarse 7x7 feature map, which doesn't have the fine-grained detail needed for pixel-perfect accuracy.
*   **The Fix in Later Versions:** Features like multi-scale predictions and better backbone networks helped later versions produce much tighter and more accurate bounding boxes.

These limitations aren't failures; they are the clues that told researchers what to fix next. Every weakness of YOLOv1 became a feature in YOLOv2, v3, and beyond.

---

## A Decade of YOLO: From v1 to v10

YOLOv1's weaknesses weren't failures; they were a roadmap. Each problem became a challenge for the next generation of researchers to solve. This kicked off a decade of rapid innovation.

### Key Leaps Forward

*   **YOLOv2 (2017): The Anchor Box Revolution.** To solve the problem of weirdly shaped objects, YOLOv2 introduced **anchor boxes**. Instead of predicting boxes from scratch, the network predicts offsets from a set of pre-defined, common box shapes. It's like giving the network a set of stencils to start with, which made predictions much easier and more accurate.

*   **YOLOv3 (2018): Seeing Big and Small.** To fix the small-object problem, YOLOv3 started making predictions at **three different scales**. It's like giving the network three pairs of glasses: one for seeing fine details up close (detecting small objects), one for medium shots, and one for the big picture. This dramatically improved its ability to handle scenes with objects of various sizes.

*   **YOLOv4 & v5 (2020): The "Kitchen Sink" Era.** These versions were all about optimization. They threw in a "bag of freebies" (clever training techniques that cost no speed) and a "bag of specials" (small tweaks that give a big accuracy boost for a tiny speed cost). YOLOv5, in particular, was a massive hit. It was implemented in PyTorch (making it super user-friendly) and came in different sizes (from Nano for mobile phones to XLarge for cloud servers).

### YOLOv10 (2024): The End-to-End Dream, Realized

After years of incremental improvements, YOLOv10 arrived with a truly radical idea: what if we could finally get rid of NMS? This wasn't just a small tweak; it was a fundamental redesign of both the training process and the architecture.

![YOLOv10 Architecture](/v10.png)
*Figure: The YOLOv10 architecture, featuring a Path Aggregation Network (PAN) neck and two specialized prediction heads.*

#### The NMS-Free Breakthrough: A Tale of Two Teachers

The problem with NMS is that it's a slow, separate step. YOLOv10 gets rid of it by using two prediction "heads" during training, like having two different teachers working together.

1.  **The Brainstormer (One-to-Many Head):** This teacher is encouraging and creative. For each object, it tells the network, "Good job! You found the dog here, and here, and also here." It matches one real object to *multiple* of the network's predictions. This provides a rich, diverse learning signal.

2.  **The Decider (One-to-One Head):** This teacher is strict and precise. For each object, it says, "There is only one correct answer. This is it." It matches one real object to *exactly one* prediction. This is the head we'll use during inference.

To make sure both teachers are on the same page, they use a **Consistent Matching Metric**. It's a shared grading rubric that scores a prediction based on three things:
*   Is it in the right area? (Spatial Prior)
*   Is it confident about the right class? (Classification Score)
*   Does the box fit well? (IOU)

By forcing both the Brainstormer and the Decider to use the same rubric, the strict Decider learns from the rich feedback given to the creative Brainstormer. To ensure they stay in sync, the model calculates the "supervision gap" between the two heads using a formula called the **1-Wasserstein distance**.

![Wasserstein Distance Formula](/algorithm.png)
*Figure: The 1-Wasserstein distance formula, used to measure the "gap" between the two heads' predictions.*

Don't let the math scare you. This formula is just a formal way of asking, "How much work do I need to do to make the Brainstormer's fuzzy predictions match the Decider's sharp prediction?" The model's goal during training is to minimize this distance.

By the end of training, the Decider is so good at picking the single best box that we don't need the Brainstormer or NMS at all. The result is a model that is truly end-to-end.

#### A More Efficient Architecture

YOLOv10's innovations didn't stop there. The authors performed a deep "internal audit" of the network to find and eliminate inefficiencies.

*   **Right-Sizing the "What" Department:** They found that the classification head (which answers "what is it?") was 2.5x more computationally expensive than the regression head ("where is it?"), even though getting the location right is more critical for accuracy. They redesigned the classification head to be much lighter and more efficient, like streamlining an overstaffed department.

*   **A Smarter Assembly Line:** Traditionally, YOLO would shrink the image and add features in one big, expensive step. YOLOv10 decouples this, using two smaller, specialized steps: one for changing the channels (features) and another for reducing the size. It's a more efficient assembly line that produces better results with less work.

*   **An Audit for Peak Performance:** They analyzed the network to find which stages were innovative (high-rank) and which were redundant (low-rank). They replaced the redundant, low-rank stages with a new, super-efficient "Compact Inverted Block" (CIB). For the important, high-rank stages, they invested more resources, using larger kernels and a bit of self-attention to help the model see the bigger picture.

### YOLOv10 in the Wild: A Real-World Example

The difference isn't just academic. On a real-world project to detect 8 different classes of vehicles from a live RTSP stream, the jump in performance is staggering.

**The Baseline (YOLOv8m):**
*   mAP50-95: 0.69

**The Upgrade (YOLOv10m):**
*   **mAP50-95: 0.76** (A huge 7% jump in overall accuracy!)
*   **mAP50: 0.906** (Meaning, for the standard IOU threshold, it's over 90% accurate)
*   **Speed:** A ~40% faster inference time than its YOLOv8 predecessor.

This is the magic of a decade of progress. The core idea of YOLOv1 remains, but its limitations have been systematically eliminated, leading to a model that is not only more accurate but even faster.

### Comparison: YOLOv1 vs YOLOv10

| Metric | YOLOv1 (2016) | YOLOv10 (2024) |
|--------|---------------|----------------|
| **mAP50** | 63.4% (VOC) | 90.6% (Custom Dataset) |
| **Post-processing** | NMS required | **NMS-free** |
| **Max Objects** | 49 (hard limit) | Virtually unlimited |
| **Small Objects** | Struggles badly | Excellent |
| **Training** | Single-head | **Dual-head** (o2m + o2o) |
| **Ease of Use** | C-based, complex | PyTorch, user-friendly |

## Conclusion: The Legacy of "You Only Look Once"

YOLOv1 was more than just a model; it was a philosophical shift. It taught the world that object detection could be framed as a single, elegant regression problem. It prioritized speed and simplicity, opening the door to applications that were previously unimaginable.

From the clunky but brilliant v1 to the sleek, end-to-end v10, the journey of YOLO is a perfect story of scientific progress: a great idea, honestly acknowledged limitations, and a decade of relentless, community-driven effort to build something better. The next time you see a self-driving car navigate a busy street, you'll know it stands on the shoulders of a giant—an idea that dared to just look once.

---

---

## Under the Hood: A PyTorch Implementation

We've talked a lot about the concepts, but what does this look like in practice? For those who like to get their hands dirty, this section provides a simplified PyTorch implementation of YOLOv1.

Don't worry if you're not a coder—feel free to skip this part! But if you're curious to see how the ideas we've discussed translate into actual code, this is for you. We'll build the model architecture, the loss function, and the dataset loader.

### 1. Model Architecture

The YOLO network uses a ResNet34 backbone (pretrained on ImageNet) followed by detection layers:

```python
import torch
import torch.nn as nn
import torchvision

class YOLOV1(nn.Module):
    """
    YOLOv1 Implementation using ResNet34 backbone

    Args:
        img_size: Input image size (448x448)
        num_classes: Number of classes (20 for Pascal VOC)
        model_config: Configuration dict with S, B, and architectural params

    Output:
        Tensor of shape (batch_size, S, S, 5*B + C)
    """
    def __init__(self, img_size, num_classes, model_config):
        super(YOLOV1, self).__init__()
        self.img_size = img_size
        self.S = model_config['S']  # Grid size (7x7)
        self.B = model_config['B']  # Boxes per cell (2)
        self.C = num_classes  # Number of classes (20)

        # Load pretrained ResNet34 backbone (trained on ImageNet 224x224)
        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )

        # Feature extraction layers (before FC layers)
        self.features = nn.Sequential(
            backbone.conv1,    # 7x7 conv, stride 2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,   # ResNet blocks
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,   # Output: 512 channels
        )

        # Detection head: 3 conv layers for feature refinement
        yolo_conv_channels = model_config['yolo_conv_channels']  # 1024
        leaky_relu_slope = model_config['leaky_relu_slope']  # 0.1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, yolo_conv_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(yolo_conv_channels),
            nn.LeakyReLU(leaky_relu_slope),

            nn.Conv2d(yolo_conv_channels, yolo_conv_channels, 3,
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(yolo_conv_channels),
            nn.LeakyReLU(leaky_relu_slope),

            nn.Conv2d(yolo_conv_channels, yolo_conv_channels, 3,
                     padding=1, bias=False),
            nn.BatchNorm2d(yolo_conv_channels),
            nn.LeakyReLU(leaky_relu_slope)
        )

        # Final 1x1 conv to get S*S*(5B+C) output
        self.final_conv = nn.Conv2d(yolo_conv_channels, 5 * self.B + self.C, 1)

    def forward(self, x):
        # x: (batch, 3, 448, 448)
        out = self.features(x)      # (batch, 512, 14, 14)
        out = self.conv_layers(out)  # (batch, 1024, 7, 7)
        out = self.final_conv(out)   # (batch, 30, 7, 7)

        # Permute to (batch, S, S, 5B+C)
        out = out.permute(0, 2, 3, 1)  # (batch, 7, 7, 30)
        return out
```

### 2. Loss Function

The complete YOLOv1 loss with all three components:

```python
import torch
import torch.nn as nn

def iou(box1, box2):
    """
    Calculate Intersection over Union between two sets of boxes.

    Args:
        box1, box2: Tensors of shape (..., 4) in format (x1, y1, x2, y2)

    Returns:
        iou: Tensor of shape (...) with IOU values
    """
    # Calculate areas
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # Find intersection rectangle
    x_topleft = torch.max(box1[..., 0], box2[..., 0])
    y_topleft = torch.max(box1[..., 1], box2[..., 1])
    x_bottomright = torch.min(box1[..., 2], box2[..., 2])
    y_bottomright = torch.min(box1[..., 3], box2[..., 3])

    # Calculate intersection area (clamp to handle non-overlapping boxes)
    intersection = (x_bottomright - x_topleft).clamp(min=0) * \
                   (y_bottomright - y_topleft).clamp(min=0)

    # Calculate union and IOU
    union = area1.clamp(min=0) + area2.clamp(min=0) - intersection
    iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero
    return iou

class YOLOLoss(nn.Module):
    """
    YOLOv1 Loss Function: The Teacher's Grading Rubric
    """
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0    # Rule #1: 5x bonus points for good coordinates
        self.lambda_noobj = 0.5    # Rule #2: Half penalty for background mistakes

    def forward(self, preds, targets):
        """
        Args:
            preds: (batch, S, S, 5*B + C) - model predictions
            targets: (batch, S, S, 5*B + C) - ground truth "answer key"

        Returns:
            loss: Scalar tensor
        """
        batch_size = preds.size(0)

        # Create coordinate shift grids for converting relative → absolute coords
        xshift = torch.arange(0, self.S, device=preds.device) / float(self.S)
        yshift = torch.arange(0, self.S, device=preds.device) / float(self.S)
        yshift, xshift = torch.meshgrid(yshift, xshift, indexing='ij')
        xshift = xshift.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        yshift = yshift.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # Reshape predictions and targets: (batch, S, S, B, 5)
        pred_boxes = preds[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, 5)
        target_boxes = targets[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, 5)

        # Convert from (x_offset, y_offset, √w, √h) to (x1, y1, x2, y2) format
        def boxes_to_x1y1x2y2(boxes, xshift, yshift):
            x_center = boxes[..., 0] / self.S + xshift
            y_center = boxes[..., 1] / self.S + yshift
            width = torch.square(boxes[..., 2])   # w = (√w)²
            height = torch.square(boxes[..., 3])  # h = (√h)²

            x1 = (x_center - 0.5 * width).unsqueeze(-1)
            y1 = (y_center - 0.5 * height).unsqueeze(-1)
            x2 = (x_center + 0.5 * width).unsqueeze(-1)
            y2 = (y_center + 0.5 * height).unsqueeze(-1)
            return torch.cat([x1, y1, x2, y2], dim=-1)

        pred_boxes_xyxy = boxes_to_x1y1x2y2(pred_boxes, xshift, yshift)
        target_boxes_xyxy = boxes_to_x1y1x2y2(target_boxes, xshift, yshift)

        # Calculate IOU between predicted and target boxes
        iou_pred_target = iou(pred_boxes_xyxy, target_boxes_xyxy)

        # Find responsible box (the one with highest IOU with ground truth)
        max_iou, max_iou_idx = iou_pred_target.max(dim=-1, keepdim=True)
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)

        # Create mask for responsible boxes
        box_indices = torch.arange(self.B, device=preds.device).reshape(1, 1, 1, self.B)
        box_indices = box_indices.expand_as(max_iou_idx)
        is_responsible_box = (max_iou_idx == box_indices).long()

        # Object indicator: 1 if cell contains object, 0 otherwise
        obj_indicator = targets[..., 4:5]  # Shape: (batch, S, S, 1)

        # Indicator for responsible boxes in cells with objects
        responsible_obj_indicator = is_responsible_box * obj_indicator

        # --- 1. GEOMETRY GRADE (Localization Loss) ---
        x_loss = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        y_loss = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        w_loss = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2  # Loss on √w
        h_loss = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2  # Loss on √h

        localization_loss = self.lambda_coord * (
            (responsible_obj_indicator * x_loss).sum() +
            (responsible_obj_indicator * y_loss).sum() +
            (responsible_obj_indicator * w_loss).sum() +
            (responsible_obj_indicator * h_loss).sum()
        )

        # --- 2. SELF-AWARENESS GRADE (Confidence Loss) ---
        # For cells that DO contain an object
        obj_conf_loss = ((pred_boxes[..., 4] - max_iou) ** 2 *
                        responsible_obj_indicator).sum()

        # For cells that DO NOT contain an object
        no_obj_indicator = 1 - responsible_obj_indicator
        noobj_conf_loss = self.lambda_noobj * (
            (pred_boxes[..., 4] ** 2 * no_obj_indicator).sum()
        )

        # --- 3. MULTIPLE CHOICE GRADE (Classification Loss) ---
        class_preds = preds[..., 5*self.B:]
        class_targets = targets[..., 5*self.B:]
        class_loss = ((class_preds - class_targets) ** 2 * obj_indicator).sum()

        # --- FINAL GRADE (Total Loss) ---
        total_loss = (localization_loss + obj_conf_loss +
                     noobj_conf_loss + class_loss) / batch_size

        return total_loss
```

### 3. Dataset & Target Encoding

Converting Pascal VOC annotations to YOLO format:

```python
import torch
import albumentations as alb
import cv2
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    """Pascal VOC Dataset with YOLO target encoding"""

    def __init__(self, split='train', img_size=448, S=7, B=2, C=20):
        self.split = split
        self.img_size = img_size
        self.S = S  # Grid size
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes

        # Data augmentation for training
        self.transforms = {
            'train': alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.Affine(scale=(0.8, 1.2),
                          translate_percent=(-0.2, 0.2)),
                alb.ColorJitter(brightness=(0.8, 1.2),
                               saturation=(0.8, 1.2)),
                alb.Resize(self.img_size, self.img_size)
            ], bbox_params=alb.BboxParams(format='pascal_voc',
                                          label_fields=['labels'])),
            'test': alb.Compose([
                alb.Resize(self.img_size, self.img_size)
            ], bbox_params=alb.BboxParams(format='pascal_voc',
                                          label_fields=['labels']))
        }

        # Load Pascal VOC annotations...
        # (XML parsing code omitted for brevity)

    def __getitem__(self, index):
        # Load image and annotations
        img_info = self.images_info[index]
        img = cv2.imread(img_info['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = [det['bbox'] for det in img_info['detections']]  # (x1,y1,x2,y2)
        labels = [det['label'] for det in img_info['detections']]

        # Apply augmentations
        transformed = self.transforms[self.split](
            image=img, bboxes=bboxes, labels=labels
        )
        img = transformed['image']
        bboxes = torch.tensor(transformed['bboxes'])
        labels = torch.tensor(transformed['labels'])

        # Normalize image to [0, 1] and apply ImageNet normalization
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()

        # --- Create the YOLO "Answer Key" Tensor ---
        target_dim = 5 * self.B + self.C
        yolo_target = torch.zeros(self.S, self.S, target_dim)

        h, w = img.shape[:2]
        cell_size = h // self.S

        if len(bboxes) > 0:
            # Convert (x1, y1, x2, y2) → (x_center, y_center, width, height)
            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            box_center_x = bboxes[:, 0] + 0.5 * box_width
            box_center_y = bboxes[:, 1] + 0.5 * box_height

            # Determine which grid cell each object belongs to
            grid_i = torch.floor(box_center_x / cell_size).long()
            grid_j = torch.floor(box_center_y / cell_size).long()

            # Compute relative coordinates (the "directions")
            box_x_offset = (box_center_x - grid_i * cell_size) / cell_size
            box_y_offset = (box_center_y - grid_j * cell_size) / cell_size

            # Normalize width and height to image size
            box_w_norm = box_width / w
            box_h_norm = box_height / h

            # Fill YOLO target tensor
            for idx in range(len(bboxes)):
                # Assign same target to all B boxes (model picks responsible one)
                for b in range(self.B):
                    s = 5 * b
                    yolo_target[grid_j[idx], grid_i[idx], s] = box_x_offset[idx]
                    yolo_target[grid_j[idx], grid_i[idx], s+1] = box_y_offset[idx]
                    yolo_target[grid_j[idx], grid_i[idx], s+2] = box_w_norm[idx].sqrt()
                    yolo_target[grid_j[idx], grid_i[idx], s+3] = box_h_norm[idx].sqrt()
                    yolo_target[grid_j[idx], grid_i[idx], s+4] = 1.0  # Confidence

                # One-hot encode class
                label = int(labels[idx])
                yolo_target[grid_j[idx], grid_i[idx], 5*self.B + label] = 1.0

        return img_tensor, yolo_target
```

### 4. Training Loop

Complete training code with proper hyperparameters:

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

# Initialize model, loss, and dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLOV1(img_size=448, num_classes=20, model_config={
    'S': 7, 'B': 2, 'yolo_conv_channels': 1024, 'leaky_relu_slope': 0.1
}).to(device)

criterion = YOLOLoss(S=7, B=2, C=20)

train_dataset = VOCDataset(split='train', img_size=448)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Optimizer: SGD with momentum (as per paper)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Learning rate schedule: reduce at epochs [75, 105]
# Paper uses warm-up from 1e-3 to 1e-2 for first epochs, then steps down
scheduler = MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)

# Training loop
num_epochs = 135  # As per paper
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
```

### 5. Inference with NMS

Post-processing predictions to get final detections:

```python
def convert_predictions_to_boxes(predictions, S=7, B=2, C=20,
                                conf_threshold=0.2, nms_threshold=0.5):
    """
    Convert YOLO predictions to bounding boxes with NMS.

    Args:
        predictions: (S, S, 5*B + C) tensor
        conf_threshold: Minimum confidence to keep box
        nms_threshold: IOU threshold for NMS

    Returns:
        boxes: (N, 4) tensor in (x1, y1, x2, y2) format
        scores: (N,) confidence scores
        labels: (N,) class labels
    """
    predictions = predictions.reshape(S, S, 5*B + C)

    # Get class predictions (same for all boxes in a cell)
    class_probs, class_labels = predictions[..., 5*B:].max(dim=-1)

    # Create coordinate shift grid
    shifts_x = torch.arange(S, device=predictions.device) / float(S)
    shifts_y = torch.arange(S, device=predictions.device) / float(S)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

    all_boxes = []
    all_scores = []
    all_labels = []

    # Process each of B boxes per cell
    for b in range(B):
        # Extract box parameters
        x_offset = predictions[..., b*5 + 0]
        y_offset = predictions[..., b*5 + 1]
        w = predictions[..., b*5 + 2]
        h = predictions[..., b*5 + 3]
        conf = predictions[..., b*5 + 4]

        # Convert to absolute coordinates
        x_center = (x_offset / S + shifts_x)
        y_center = (y_offset / S + shifts_y)
        width = torch.square(w)   # w = √w_pred²
        height = torch.square(h)

        # Convert to (x1, y1, x2, y2) format
        x1 = (x_center - 0.5 * width).reshape(-1, 1)
        y1 = (y_center - 0.5 * height).reshape(-1, 1)
        x2 = (x_center + 0.5 * width).reshape(-1, 1)
        y2 = (y_center + 0.5 * height).reshape(-1, 1)
        boxes = torch.cat([x1, y1, x2, y2], dim=-1)

        # Compute class-specific confidence scores
        scores = conf.reshape(-1) * class_probs.reshape(-1)
        labels = class_labels.reshape(-1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    # Concatenate all boxes
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Confidence thresholding
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Apply NMS per class
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(labels):
        class_indices = labels == class_id
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]

        # NMS (using torchvision)
        keep_indices = torch.ops.torchvision.nms(
            class_boxes, class_scores, nms_threshold
        )

        # Mark these boxes as kept
        class_keep_indices = torch.where(class_indices)[0][keep_indices]
        keep_mask[class_keep_indices] = True

    final_boxes = boxes[keep_mask]
    final_scores = scores[keep_mask]
    final_labels = labels[keep_mask]

    return final_boxes, final_scores, final_labels

# Example usage
model.eval()
with torch.no_grad():
    img_tensor = ... # Load and preprocess image
    predictions = model(img_tensor.unsqueeze(0))[0]  # Remove batch dim

    boxes, scores, labels = convert_predictions_to_boxes(
        predictions, conf_threshold=0.2, nms_threshold=0.5
    )

    # boxes: (N, 4) in normalized 0-1 coordinates
    # Multiply by image dimensions to get pixel coordinates
```

### Key Implementation Details:

1. **√w and √h**: The model predicts square root of width/height to balance loss across different box sizes
2. **Relative Coordinates**: x,y offsets are relative to grid cell top-left corner
3. **Responsible Box Selection**: During training, only the box with highest IOU with GT is penalized for coordinates
4. **Class Probabilities**: Shared across all B boxes in a cell (limitation of v1)
5. **Lambda Weighting**: λ_coord=5 to emphasize localization, λ_noobj=0.5 to de-emphasize empty cells

This implementation closely follows the original paper and can achieve ~63% mAP on Pascal VOC 2007 after 135 epochs of training.

---

### Optimization Techniques

#### Precision Options

| Precision | Speed | Accuracy | Memory | Best For |
|-----------|-------|----------|--------|----------|
| **FP32** (default) | 1× | 100% (baseline) | 4 bytes/param | Training, research |
| **FP16** (half precision) | **2-3×** | 99.5% | 2 bytes/param | Production (GPU) |
| **INT8** (quantization) | **4×** | 97-99% | 1 byte/param | Edge devices |

**INT8 Quantization** (PyTorch):
```python
import torch.quantization

model = YOLOV1(...)
model.eval()

# Quantization-aware training (best accuracy)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train for a few epochs...
model_quantized = torch.quantization.convert(model_prepared)

# Result: 4× smaller, 2-4× faster on CPU
```

#### Layer Fusion
TensorRT automatically fuses operations:
```
Conv → BatchNorm → ReLU  →  Single fused kernel
                            (3× fewer memory transfers)
```

#### Dynamic Tensor Memory
- Reuses memory buffers across layers
- Reduces GPU memory usage by 30-50%

### 3. Hardware-Specific Deployment

#### NVIDIA GPU (RTX 3060/4090)
**Recommendation**: TensorRT with FP16

```bash
# Export PyTorch → ONNX → TensorRT
python export.py --weights best.pt --format onnx
trtexec --onnx=yolov1.onnx --saveEngine=yolov1_fp16.trt --fp16
```

**Expected Performance**:
- RTX 4090: 180 fps (FP16), 90 fps (FP32)
- RTX 3060: 120 fps (FP16), 60 fps (FP32)
#### Edge Devices (Jetson, Raspberry Pi)
**Recommendation**: TensorRT INT8 or TFLite

**NVIDIA Jetson Orin**:
```bash
# INT8 quantization for Jetson
trtexec --onnx=yolov1.onnx \
        --int8 \
        --workspace=2048 \
        --saveEngine=yolov1_int8.trt
```
Performance: 120 fps (INT8), 45 fps (FP16)

---

## Conclusion

YOLOv1 fundamentally changed object detection by proving that:
1. **Single-stage detection works**: No need for complex multi-stage pipelines
2. **Real-time is possible**: 45 fps without sacrificing too much accuracy
3. **Simplicity is powerful**: Unified architecture and loss function
4. **Context matters**: Seeing the whole image reduces false positives

While YOLOv1 had limitations (small objects, crowded scenes, unusual aspect ratios), it established the foundation for a family of detectors that continues to push the boundaries of speed and accuracy.

The evolution from v1 to v10 shows continuous improvement while maintaining the core philosophy: **You Only Look Once** — simple, fast, and effective object detection.

---

## References

- [You Only Look Once: Unified, Real-Time Object Detection (2016)](https://arxiv.org/abs/1506.02640)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [ImageNet](http://www.image-net.org/)
- [PyTorch Implementation](https://github.com/samitmohan/YOLOv1/tree/master/implementation)
- [How CNN works](https://cs231n.stanford.edu/)


---


