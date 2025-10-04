

### Dance Movement Analysis Server

A cloud-based AI/ML server that processes short-form dance videos to detect and summarize standard dance poses. This project is built with Python, MediaPipe, and Flask, containerized with Docker, and deployed on a Google cloud platform.



#### Features

* **Pose Detection**: Analyzes video files to detect 33 key body landmarks in each frame.
* **Movement Classification**: Identifies pre-defined dance poses like "T-Pose" and "Arms Up" based on joint angles.
* **API Interface**: Provides a simple REST API endpoint to upload videos and receive a JSON summary of the analysis.
* **Containerized \& Scalable**: Packaged with Docker for consistent, reliable deployment on any cloud environment.



### Setup and Local Development

To run this project on your local machine, you'll need Git and Docker Desktop installed.



###### 1\. Clone the Repository

**Bash:**

git clone <your-github-repo-url>

cd <your-repo-name>



###### 2\. Build the Docker Image

This command builds the Docker image from the Dockerfile, installing all system and Python dependencies. The first build will take several minutes.



**Bash:**

docker build -t dance-analyzer .



###### 3\. Run the Docker Container

This command starts the application inside a container and maps port 5000 on your machine to the container's port 5000.



**Bash:**

docker run -p 5000:5000 dance-analyzer

The server is now running at http://localhost:5000.



###### **API Usage**



**Endpoint**: POST /analyze

Uploads a short video file for pose analysis.



**Request:**

* Method: POST
* Body: multipart/form-data
* Field: file - The video file (.mp4, .mov, etc.) to be analyzed.



###### Example curl Command:



**Bash:**

curl -X POST -F "file=@/path/to/your/video.mp4" http://localhost:5000/analyze



**Successful Response (200 OK):**

The server returns a JSON object summarizing the detected poses and the number of frames each pose was identified in.



**JSON**



{

&nbsp; "duration\_s": 15.33,

&nbsp; "pose\_summary": \[

&nbsp;   {

&nbsp;     "pose": "arms\_up",

&nbsp;     "frames\_detected": 216

&nbsp;   },

&nbsp;   {

&nbsp;     "pose": "standing",

&nbsp;     "frames\_detected": 236

&nbsp;   },

&nbsp;   {

&nbsp;     "pose": "t\_pose",

&nbsp;     "frames\_detected": 6

&nbsp;   }

&nbsp; ],

&nbsp; "processed\_frames": 460,

&nbsp; "total\_frames": 460,

&nbsp; "video\_path": "uploads/your\_video.mp4"

}



**Error Response (400 Bad Request):**



**JSON**



{

&nbsp; "error": "No file part in the request"

}



### Thought Process \& Design Choices

My approach to this project was to build a robust, production-ready application by separating concerns and choosing the right tools for each task.



**1.Core Logic (analyzer.py):** I chose **MediaPipe** for pose estimation due to its high accuracy, performance, and detailed 33-point landmark model. The core analysis is based on calculating the angles between key body joints, which is a computationally efficient and effective way to define and detect specific poses.



**2.API \& Server (app.py):** **Flask** was selected for its simplicity and lightweight nature, making it ideal for a single-endpoint microservice. During development, it became clear that the built-in Flask development server was not stable enough to handle video file uploads and long-running analysis tasks in a container. I replaced it with **Gunicorn**, a production-grade **WSGI server**, which resolved timeout and stability issues.



**3.Containerization (Dockerfile):** From the beginning, the project was designed to be containerized with Docker. This decouples the application from the underlying infrastructure, solving the "it works on my machine" problem and guaranteeing a consistent environment from local development to cloud deployment. The Dockerfile starts with a minimal **python:3.9-slim** base image and adds only the necessary system dependencies (libgl1, libglib2.0-0) to support OpenCV, keeping the final image as small as possible.



**4.Debugging \& Deployment:** The journey to a stable cloud deployment involved several debugging cycles. Early crashes in the container were traced back to missing system libraries and out-of-memory errors on small cloud VMs. This was solved by adding the dependencies to the Dockerfile and selecting an appropriately sized VM **(e2-standard-2 on GCP)**. Later, Connection reset errors were diagnosed as **Gunicorn** worker timeouts, which was fixed by adding a -**-timeout** flag to the startup command. This iterative debugging process was crucial for creating a reliable final product



### Connecting with Callus's Vision

This project was designed and executed to directly align with Callus's vision for a modern AI/ML Server Engineer, which emphasizes end-to-end ownership and the deployment of practical, scalable services.



Rather than focusing solely on the Python script for movement analysis, I embraced a holistic, full-stack approach. The project demonstrates the complete lifecycle of a modern AI service, starting with core feature development in Python and unit testing to ensure accuracy and reliability.



The decision to containerize the application with 



Docker was central to the design. This choice ensures that the service is portable, reproducible, and ready for scalable deployment, reflecting a commitment to modern DevOps principles that are critical for cloud environments. By exposing the analysis feature through a robust 



Flask/Gunicorn API, the project transforms a standalone AI model into a practical, usable service that can be integrated into larger applications—a key step in creating functional AI products.



Finally, deploying the container to 



Google Cloud Platform showcases the ability to operate within a production-grade cloud environment. This entire process—from a local script to a live, documented public endpoint—is a direct reflection of the full-stack and DevOps-oriented skill set required to not just build AI models, but to successfully deliver them as robust and scalable cloud services.









