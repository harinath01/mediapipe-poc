import {
    FaceDetector,
    FilesetResolver,
    FaceLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


let faceDetector;
let faceLandMarker;
const video = document.getElementById("webcam");
const anomalyList = document.getElementById("anomalyList");

// Load face-api models

// Start webcam stream
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            
            initializeMediaPipes().then(() => { 
                setInterval(() => {
                    detectFace();
                }, 3000);
            })
        }) 
        .catch(err => console.error("Error accessing webcam:", err));
}

const initializeMediaPipes = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
            delegate: "GPU"
        },
        runningMode: "IMAGE"
    });

    faceLandMarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: false,
        runningMode: "IMAGE",
        numFaces: 1
    });
};

function addAnomaly(type) {
    const timestamp = new Date().toLocaleTimeString();
    const li = document.createElement('li');
    li.textContent = `${timestamp}: ${type}`;
    anomalyList.appendChild(li);

    anomalyContainer.scrollTop = anomalyContainer.scrollHeight;
}

function createCanvasElement(videoElement) {
    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth || videoElement.naturalWidth;
    canvas.height = videoElement.videoHeight || videoElement.naturalHeight;
    const context = canvas.getContext("2d");
    if (context) {
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        return canvas;
    } else {
        throw new Error("2D context for canvas is null or undefined.");
    }
}

// Face detection and anomaly detection
async function detectFace() {
    const canvas = createCanvasElement(video);
    const result = await faceDetector.detect(canvas);
    const detections = result.detections
    
    if (detections.length === 0) {
        addAnomaly('No person detected');
    } else if (detections.length > 1) {
        addAnomaly('Multiple persons detected');
    } else {
        addAnomaly('Person detected');
        const faceMeshResult = await faceLandMarker.detect(canvas);

        if (isFaceTurned(faceMeshResult.faceLandmarks[0], canvas.width)) { 
            addAnomaly('Looking Away');
        }
    }
}

function isFaceTurned(landmarks, imageWidth) {
    const nose = landmarks[1]; // Nose tip
    const leftEyeX = landmarks[33].x * imageWidth; // Left eye outer corner
    const rightEyeX = landmarks[263].x * imageWidth; // Right eye outer corner

    // Calculate the midpoint between the left and right eye x-coordinates in pixels
    const eyesMidX = (leftEyeX + rightEyeX) / 2;

    // Get the nose x-position in pixels
    const noseX = nose.x * imageWidth;

    // Get jawline positions (adjust landmark indices as necessary)
    const leftJawX = landmarks[5].x * imageWidth; // Left jaw corner
    const rightJawX = landmarks[11].x * imageWidth; // Right jaw corner

    // Calculate jaw midpoint
    const jawMidX = (leftJawX + rightJawX) / 2;

    // Calculate offsets
    const noseOffset = Math.abs(noseX - eyesMidX);
    const jawOffset = Math.abs(jawMidX - eyesMidX);

    const threshold = imageWidth * 0.1; // Adjust this percentage based on testing

    console.log("Left Eye:", leftEyeX);
    console.log("Right Eye:", rightEyeX);
    console.log("Nose:", noseX);
    console.log("Jaw Midpoint:", jawMidX);
    console.log("Nose Offset:", noseOffset);
    console.log("Jaw Offset:", jawOffset);
    console.log("Threshold:", threshold);
    console.log("========================");

    // Combine the offsets for a more comprehensive check
    return noseOffset > threshold || jawOffset > threshold;
}

startWebcam()