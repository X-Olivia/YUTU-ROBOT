# Development Log · Week 1

## Hardware Procurement Status
This week completed the procurement of chassis and inventory management related hardware, details as follows:

- **Chassis Related Hardware**
  - Chassis shell
  - 14-inch touchscreen
  - Jetson Orin Nano
  - M10P LiDAR
  - RC remote control
  - Automatic charging module
  - iFLYTEK streaming microphone array voice module
  - Depth camera

- **Inventory Management Related Hardware**
  - Wide-angle camera
  - Weight sensor
  - Reed sensor
  - Raspberry Pi 4B

- **Additional Equipment**
  - Raspberry Pi cooling fan
  - Various adapter cables
  - Solid state drive
  - Xiaomi test phone

## Expense Record
- Chassis hardware: ¥13,354  
- Raspberry Pi fan: ¥30  
- Adapter cables: ¥16 + ¥49  
- Wide-angle camera: ¥213  
- Reed sensor: ¥22  
- Weight sensor: ¥25.4  
- Display screen: ¥261  
- Solid state drive: ¥135  
- Xiaomi test phone: ¥288  

**Total**: ¥14,393.4

## This Week's Development Focus
1. **Overall Architecture Design**
   - Organized the overall architecture of robot, cloud (deprecated), and App (data flow, storage and synchronization mechanisms, communication strategies).
   - Clarified the division of responsibilities between the robot body (Raspberry Pi / Jetson) and the app.

2. **Inventory Management · Image Recognition Development**
   - Set up preliminary testing environment using wide-angle camera.
   - Started YOLO-based object detection experiments, target categories: day/night sanitary pads, panty liners, tampons, pain relief medication.
   - Explored **vision + weight sensor** redundant design to solve "empty packaging" recognition problem.  
   - Designed data collection and annotation workflow, built small-scale dataset for training.  
For details, see:
---

