# Computer Vision Development Log Week 1

## Initial Design Concept
The initial inventory recognition solution was a combination of **weight sensor + camera**:
- **When drawer opens**: Weigh once and take a photo.  
- **When drawer closes**: Weigh again and take another photo.  
- If there's a significant difference between the two weighings → Compare images before and after opening/closing, analyze items added or removed.  

### Problems Encountered
1. **Unable to identify specific product models**  
   - Can only detect that items have changed, but cannot distinguish between day/night sanitary pads, overnight pants, etc.  
2. **Limited camera viewing angle**  
   - To cover the entire drawer, the camera needs to be placed at a high position, causing low device space utilization.  
   - Multiple cameras are feasible but result in complex structure, high cost, and large space occupation.  

---

## Preliminary Experiments
- Created a small database using **Roboflow**.  
- Model: `YOLOv11`  
- Results:  
  - Can perform basic recognition of packaging boxes, but difficult to distinguish similar-looking products like day/night sanitary pads, tampons, overnight pants, etc.  
  - After introducing **OCR (Optical Character Recognition)**, static image classification performance improved significantly.  

---

## New Solution Exploration
To address the above issues, tried two types of improvement approaches:

### **Solution 1: Weight Sensor Only**
- Divide the drawer into multiple zones, with each zone predefined for one type of product.  
- Track item usage based on weight changes in each zone.  
- **Advantages**: Simple implementation, stable and reliable.  
- **Disadvantages**: Users must follow fixed placement rules, poor flexibility.

---

### **Solution 2: Vision + Weight Combination**
- **User Registration**: Register category when first placing products (no zoning, free placement).  
- **Camera Position**: Install at the upper edge outside the drawer, facing the user, able to see the entire process of items being taken/placed.  
- **Recognition Logic**:  
  1. Drawer opens → Start recording.  
  2. Drawer closes → Stop recording.  
  3. If weight difference exists → Analyze video, track items with the largest hand movement range; match with registered products, then combine with weight difference to confirm usage.  
  4. If no difference → Keep video in short-term cache then auto-delete.  

- **Implementation Progress**:  
  Used the **YOLOv8 + BYTETracker + Supervision** combination:  
  - **YOLOv8**: Responsible for frame-by-frame object detection, identifying item categories and positions in the image.  
  - **BYTETracker**: Based on YOLOv8 detection results, performs **object tracking** between video frames, ensuring continuous tracking of the same item's movement trajectory.  
  - **Supervision**: Mainly used for processing video frames, drawing bounding boxes, labels and trajectory lines, combined with LineZone for crossing line analysis, annotating detection and tracking results to video, finally saving and displaying output video.  

  Currently shows good experimental results with small datasets + specific scenarios.  
  Next step is to expand the dataset and verify robustness in more usage scenarios.  

---

## Attempted Methods (Abandoned)
- Tried **Vision + Weight + OCR frame capture solution**:  
  - Capture clear frames of moving objects in video, use OCR to read packaging text to confirm categories.  
  - Actual results were unsatisfactory: frequently encountered hand occlusion, blurred frames, and insufficient accuracy.  

---

## Reflection: Analysis of Failed Methods
Looking back at methods attempted but failed during development, analyzing failure reasons:

1. **Static Photo + Simple Comparison Solution Failure Reasons**:
   - For small-sized robots, this solution reduced space utilization
   - Lack of temporal information, unable to understand user's true operational intent
   - Extremely restrictive for user item placement

2. **OCR Frame Capture Recognition Solution Failure Reasons**:
   - Frequent hand occlusion, limited opportunities to obtain clear packaging text
   - Unstable video frame quality, motion blur affects OCR accuracy
   - Large angle variations in packaging text, OCR recognition rate drops significantly in real scenarios
   - High processing latency, unable to meet real-time requirements

---

## Current Direction
- **Package Matching Solution**:  
  - No longer rely on OCR, but utilize packaging color and appearance differences for matching.  
  - For most products, packaging shows obvious color differences, making matching highly feasible.  
- **Next Week's Plan**:  
  - Conduct matching experiments, verify accuracy and stability in more complex environments.  
  - Expand dataset to cover different lighting, angles, and user operation methods.  

---

**Summary**: Computer vision development has gradually evolved from "static snapshot + OCR" to "video tracking + package matching + weight redundancy", better meeting actual usage requirements in terms of flexibility and accuracy. Practice more on seemingly impossible things - if you fail, you better understand the reasons for failure; if you succeed, that's even better.
