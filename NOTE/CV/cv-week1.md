# Computer Vision Development Log Week 1

## Initial Design Approach
The initial inventory recognition solution was a combination of **weight sensor + camera**:
- **When drawer opens**: Weigh once and take a photo.  
- **When drawer closes**: Weigh again and take another photo.  
- If there's a significant difference in weight → Compare images before and after opening/closing to analyze items added or removed.  

### Problems Encountered
1. **Unable to identify specific product models**  
   - Can only detect item changes, but cannot distinguish between day/night sanitary pads, adult diapers, etc.  
2. **Limited camera viewing angle**  
   - To cover the entire drawer, camera needs to be placed at a high position, reducing device space utilization.  
   - Using multiple cameras is feasible but increases structural complexity, cost, and space requirements.  

---

## Preliminary Experiments
- Used **Roboflow** to create a small custom database.  
- Model: `YOLOv11`  
- Results:  
  - Can perform basic recognition on packaging boxes, but difficult to distinguish similar-looking products like day/night sanitary pads, tampons, adult diapers, etc.  
  - After introducing **OCR (text recognition)**, static image classification improved significantly.  

---

## New Solution Exploration
To address the above issues, tried two types of improvement approaches:

### **Solution 1: Weight Sensor Only**
- Divide the drawer into multiple regions, with each region predefined for one type of product.  
- Track item usage based on weight changes in each region.  
- **Advantages**: Simple implementation, stable and reliable.  
- **Disadvantages**: Users must follow fixed placement rules, poor flexibility.

---

### **Solution 2: Vision + Weight Combination**
- **User Registration**: Register item categories when first placing products (no zone restrictions, free placement).  
- **Camera Position**: Installed at the upper edge outside the drawer, facing the user, can observe the entire process of item retrieval and placement.  
- **Recognition Logic**:  
  1. Drawer opens → Start recording.  
  2. Drawer closes → Stop recording.  
  3. If weight difference exists → Analyze video, track items with the largest hand movement range; match with registered products, combined with weight difference to confirm usage.  
  4. If no difference → Keep video in short-term cache then auto-delete.  

- **Implementation Progress**:  
  Used **YOLOv8 + BYTETracker + Supervision** combination:  
  - **YOLOv8**: Responsible for frame-by-frame object detection, identifying item categories and positions in the frame.  
  - **BYTETracker**: Based on YOLOv8 detection results, performs **object tracking** between video frames, ensuring continuous tracking of the same item's movement trajectory.  
  - **Supervision**: Mainly used for processing video frames, drawing bounding boxes, labels and trajectory lines, combined with LineZone for crossing line analysis, annotating detection and tracking trajectory results onto video, finally saving and displaying output video.  

  Currently shows good experimental results on small datasets + specific scenarios.  
  Next step requires expanding the dataset and validating robustness in more usage scenarios.  

---

## Attempted Methods (Deprecated)
- Tried **Vision + Weight + OCR frame extraction approach**:  
  - Extract clear frames of moving objects in video, use OCR to read packaging text for category confirmation.  
  - Actual results were unsatisfactory: frequently encountered hand occlusion, frame blur and other issues, insufficient accuracy.  

---

## Reflection: Analysis of Failed Methods
Reviewing methods attempted but failed during development, analyzing failure reasons:

1. **Static Photography + Simple Comparison Method Failure Reasons**:
   - For small-size robots, this approach reduced space utilization efficiency
   - Lack of temporal information, unable to understand user's actual operational intent
   - Extremely restrictive on user item placement

2. **OCR Frame Recognition Method Failure Reasons**:
   - Frequent hand occlusion, limited opportunities to obtain clear packaging text
   - Unstable video frame quality, motion blur affects OCR accuracy
   - Large variations in packaging text angles, OCR recognition rate drops significantly in real scenarios
   - High processing latency, unable to meet real-time requirements

---

## Current Direction
- **Package Matching Approach**:  
  - No longer relying on OCR, but utilizing packaging color and appearance differences for matching.  
  - For most products, packaging has obvious color differences, making matching highly feasible.  
- **Next Week's Plan**:  
  - Conduct matching experiments to verify accuracy and stability in more complex environments.  
  - Expand dataset to cover different lighting, angles, and user operation methods.  

---

**Summary**: Has evolved from "static snapshots + OCR" to "video tracking + package matching + weight redundancy", better meeting actual usage requirements in terms of flexibility and accuracy. 
