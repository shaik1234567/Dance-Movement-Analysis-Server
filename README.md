# Dance Movement Analysis Server

## Overview

This project implements a dance movement analysis system using MediaPipe and OpenCV to detect and classify various dance poses from video inputs. It provides detailed pose durations, frame counts, and timeline segmentation in JSON format via an HTTP API.

Packaged as a Docker container, this server enables easy deployment and consistent environment setup for pose analysis tasks.

---

## Features

- Multi-pose detection including standing, squat, arms_up, side_stretch, and more.
- JSON output with pose timelines, frame statistics, and pose durations.
- Automated unit testing validating detection logic using `pytest`.
- Dockerized for consistent deployment and easy usage on various platforms.

---

## Prerequisites

- Docker installed and running on your local machine.
- Video files (e.g., MP4 format) for analysis.

---

## Installation and Usage

### Build the Docker Image

Clone the repository or copy the project folder. Run in project root where Dockerfile is located:

