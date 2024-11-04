import {
    FaceDetector,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


let faceDetector;
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
    console.log(result)
    const detections = result.detections
    
    if (detections.length === 0) {
        addAnomaly('No person detected');
    } else if (detections.length > 1) {
        addAnomaly('Multiple persons detected');
    } else {
        addAnomaly('Person detected');

        if (isFaceTurned(result.detections[0])) {
            addAnomaly('Face turned');
        }
    }
}

function isFaceTurned(detection) {
    const landmarks = detection.keypoints;
    const imageWidth = detection.boundingBox.width;
    const leftEyeX = landmarks[0].x * imageWidth;
    const rightEyeX = landmarks[1].x * imageWidth;
    const jawLeftX = landmarks[4].x * imageWidth;
    const jawRightX = landmarks[5].x * imageWidth;
    const faceWidth = jawRightX - jawLeftX; 
    const faceCenterX = (jawLeftX + jawRightX) / 2;
    
    console.log("Jaw Left:", jawLeftX, "Jaw Right:", jawRightX);
    console.log("Left Eye X:", leftEyeX, "Right Eye X:", rightEyeX, "Face Center X:", faceCenterX, "Face Width:", faceWidth);

    const threshold = 0.15; // Slightly increased threshold
    console.log("Distance from Left Eye to Center:", Math.abs(leftEyeX - faceCenterX));
    console.log("Allowed Threshold:", faceWidth * threshold);
    console.log("Is Left Eye within threshold:", Math.abs(leftEyeX - faceCenterX) < faceWidth * threshold);
    console.log("Is Right Eye within threshold:", Math.abs(rightEyeX - faceCenterX) < faceWidth * threshold); 
    return (
      Math.abs(leftEyeX - faceCenterX) < faceWidth * threshold ||
      Math.abs(rightEyeX - faceCenterX) < faceWidth * threshold
    );
}

startWebcam()