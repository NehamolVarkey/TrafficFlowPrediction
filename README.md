<h1>AI Based Traffic Flow Prediction🚗</h1>
<p>The "AI Based Traffic Flow Prediction" implements a car counting system using OpenCV, designed to track vehicles in a video feed. 
The system utilizes a pre-trained MobileNetSSD model for object detection, specifically targeting cars. 
It employs centroid tracking to monitor the movement of detected cars across frames. 

The system uses the MobileNetSSD model to detect cars in each frame of the video feed. It filters out detections with a confidence threshold of 0.5 and extracts bounding box coordinates for the detected cars. 

The centroid tracker maintains the identity of vehicles by associating their centroids across consecutive frames. It calculates the centroids of the detected cars and updates their positions over time. If a car disappears from the frame for a certain number of consecutive frames (controlled by parameters maxDisappeared and maxDistance), it is considered to have left the scene, and its tracking is terminated.

The system counts the number of cars present in the scene at any given time and logs their entry time. It maintains a dictionary to store the object IDs and their corresponding entry timestamps. The count is displayed on the video feed in real-time and is also saved to a CSV file upon exiting the program.
A function for non-maximum suppression is provided to remove redundant bounding boxes and retain only the most relevant ones.

This system provides a foundational framework for developing car counting systems and can be extended or modified for various applications such as traffic monitoring, parking management, or retail analytics.</p>
