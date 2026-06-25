# AI Smart Care Cane

AI Smart Care Cane is an AI-based fall detection prototype designed for a smart care cane system.  
This project aims to detect fall-related movements using sensor data and provide real-time monitoring and emergency alerts through a mobile application.

This project was developed as part of the AI Convergence Project.

---

## Overview

Falls among older adults are a major cause of injury and require rapid detection for timely assistance.

To address this problem, this project develops a prototype system that combines:

- Sensor-based fall detection
- AI-based fall classification model
- FastAPI backend server
- SQLite database
- React-based client application

The system is designed around the concept of a smart care cane that can support elderly users by detecting abnormal movement patterns and notifying caregivers when a possible fall occurs.

---

## Key Features

- AI-based fall detection using motion sensor data
- Classification of fall and Activities of Daily Living (ADL)
- Real-time monitoring system prototype
- Emergency alert interface
- Backend API server for data processing
- SQLite database for storing sensor and user-related data
- React-based client interface

---

## Tech Stack

### Frontend

- React
- JavaScript
- HTML / CSS

### Backend

- FastAPI
- Python
- SQLite3

### AI Model

- Python
- PyTorch
- CNN-based deep learning model
- Accelerometer and gyroscope sensor data

### Database

- SQLite3

---

## System Architecture

The system consists of four main parts:

1. **IoT Side**
   - Smart care cane concept
   - Sensor data collection from motion sensors

2. **AI Model**
   - Fall detection model
   - Classification of fall and ADL data

3. **Server Side**
   - FastAPI backend server
   - Receives sensor data
   - Sends recent sensor slices to the AI model
   - Stores and retrieves data from SQLite3

4. **Client Side**
   - React-based interface
   - Displays user information
   - Shows fall detection alerts

```text
Smart Care Cane Sensor Data
            ↓
      FastAPI Server
            ↓
      SQLite3 Database
            ↓
        AI Model
            ↓
     React Client App
            ↓
   Fall Detection Alarm
