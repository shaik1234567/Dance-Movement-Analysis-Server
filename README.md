# Dance Movement Analysis Server

A cloud-based AI/ML server that processes short-form dance videos to detect and summarize dance poses using advanced computer vision. Built with Python, MediaPipe, and Flask, containerized with Docker, and deployed on Google Cloud Platform.

## Features

* **Advanced Pose Detection**: Analyzes video files to detect 33 key body landmarks in each frame using MediaPipe
* **17 Movement Classifications**: Identifies comprehensive dance poses including arms_up, t_pose, squat, jump, lunges, side stretches, leans, and more
* **Temporal Timeline Analysis**: Tracks pose sequences throughout the video with precise timestamps and duration metrics
* **Performance Statistics**: Provides pose frequency, duration, and percentage breakdowns
* **REST API Interface**: Simple endpoint to upload videos and receive detailed JSON analysis
* **Containerized & Scalable**: Packaged with Docker for consistent deployment on any cloud environment

## Detected Poses

The analyzer detects 17 different poses:

**Arms & Upper Body:**
- arms_up, arms_forward, t_pose, side_stretch_left, side_stretch_right, crossed_arms, hands_on_hips

**Lower Body & Movement:**
- squat, lunge_left, lunge_right, jump, one_leg_stand_left, one_leg_stand_right

**Body Position:**
- lean_left, lean_right, standing, no_person

## Setup and Local Development

### Prerequisites
- Git
- Docker Desktop

### 1. Clone the Repository

```bash
git clone <github-repo-url>
cd <your-repo-name>
```

### 2. Build the Docker Image

```bash
docker build -t dance-analyzer .
```

### 3. Run the Docker Container

```bash
docker run -p 5000:5000 dance-analyzer
```

The server is now running at `http://localhost:5000`

## API Usage

### Endpoint: `POST /analyze`

Uploads a video file for pose analysis.

**Request:**
- Method: `POST`
- Body: `multipart/form-data`
- Field: `file` - The video file (.mp4, .mov, etc.)

### Example:

```bash
curl -X POST -F "file=@/path/to/your/video.mp4" http://localhost:5000/analyze
```


## Thought Process & Design Choices

### 1. Core Logic (analyzer.py)

**MediaPipe** was chosen for pose estimation due to its high accuracy and detailed 33-point landmark model. The enhanced version includes:

- **Comprehensive Joint Angles**: Calculates angles for elbows, shoulders, hips, and knees
- **17 Pose Classifications**: Rule-based detection system with priority ranking (dynamic poses like jumps take precedence over static poses)
- **Temporal Analysis**: Tracks pose sequences across frames with accurate timestamps and duration calculations
- **Bilateral Support**: Detects left/right variants separately (lunges, side stretches, one-leg stands)

### 2. API & Server (app.py)

**Flask** provides a lightweight REST API, while **Gunicorn** serves as the production-grade WSGI server. Gunicorn resolved stability issues with video uploads and long-running analysis tasks that the Flask development server couldn't handle reliably.

### 3. Containerization (Dockerfile)

The project uses Docker to ensure consistent deployment across environments. The **python:3.9-slim** base image keeps the container lightweight while including necessary system dependencies (libgl1-mesa-glx, libglib2.0-0) for OpenCV support.

### 4. Debugging & Deployment

The deployment journey involved debugging missing system libraries and memory constraints. The solution required:
- Adding OpenCV dependencies to Dockerfile
- Selecting appropriate VM size **(e2-standard-2 on GCP)**
- Configuring Gunicorn with **--timeout** flags to handle long video processing

## Connecting with Callus's Vision

This project demonstrates end-to-end ownership of an AI/ML service, from core algorithm development to scalable cloud deployment. 

The implementation goes beyond a standalone Python script to deliver a complete, production-ready service. By containerizing with Docker and exposing functionality through a robust Flask/Gunicorn API, the project transforms an AI model into a practical, integrable service.

Deploying to Google Cloud Platform showcases the ability to operate in production-grade cloud environments. This complete lifecycle—from local development to a live, documented public endpoint—reflects the full-stack and DevOps-oriented skill set required to successfully deliver robust and scalable cloud services.
