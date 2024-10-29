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
    if (!landmarks || landmarks.length === 0) {
        return false;
    }

    // Let's use more points for better accuracy
    // Left side face points (from eye to jaw)
    const leftFacePoints = [
        landmarks[226], // Below eye
        landmarks[447], // Upper jaw
        landmarks[366], // Mid jaw
        landmarks[361], // Near chin
        landmarks[297]  // Chin
    ];

    // Right side face points (from eye to jaw)
    const rightFacePoints = [
        landmarks[446], // Below eye
        landmarks[227], // Upper jaw
        landmarks[137], // Mid jaw
        landmarks[132], // Near chin
        landmarks[297]  // Chin
    ];

    // Calculate the horizontal spread of points on each side
    function calculateSideSpread(points) {
        const xCoords = points.map(p => p.x);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const spread = Math.abs(maxX - minX) * imageWidth;
        return spread;
    }

    const leftSpread = calculateSideSpread(leftFacePoints);
    const rightSpread = calculateSideSpread(rightFacePoints);

    // Calculate distances from center point (nose) to each side
    const noseTip = landmarks[4];  // Nose tip landmark
    const leftMostPoint = Math.min(...leftFacePoints.map(p => p.x));
    const rightMostPoint = Math.max(...rightFacePoints.map(p => p.x));

    const leftDistance = Math.abs(noseTip.x - leftMostPoint) * imageWidth;
    const rightDistance = Math.abs(rightMostPoint - noseTip.x) * imageWidth;

    // Calculate asymmetry ratio
    const asymmetryRatio = Math.min(leftDistance, rightDistance) /
        Math.max(leftDistance, rightDistance);

    // When head is significantly turned:
    // Lower threshold for more sensitivity
    const TURN_THRESHOLD = 0.65;

    const isTurnedSignificantly = asymmetryRatio < TURN_THRESHOLD;
    const turnDirection = isTurnedSignificantly ?
        (leftDistance < rightDistance ? 'right' : 'left') : 'center';

    // Detailed logging for debugging
    console.log({
        leftSpread: leftSpread.toFixed(2),
        rightSpread: rightSpread.toFixed(2),
        leftDistance: leftDistance.toFixed(2),
        rightDistance: rightDistance.toFixed(2),
        asymmetryRatio: asymmetryRatio.toFixed(2),
        direction: turnDirection,
        isTurned: isTurnedSignificantly,
        threshold: TURN_THRESHOLD
    });

    // Alternative detection method using visibility
    // Check if certain landmarks are hidden/visible
    const leftEyeVisible = landmarks[33].z < 0;  // Left eye outer corner
    const rightEyeVisible = landmarks[263].z < 0; // Right eye outer corner

    // If one eye is significantly more visible than the other, head is definitely turned
    if (leftEyeVisible && !rightEyeVisible) {
        console.log("Right turn detected by visibility");
        return true;
    }
    if (!leftEyeVisible && rightEyeVisible) {
        console.log("Left turn detected by visibility");
        return true;
    }

    return isTurnedSignificantly;
}

startWebcam()