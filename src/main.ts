import {FaceDetector, FaceLandmarker, FaceLandmarkerResult, FilesetResolver} from "@mediapipe/tasks-vision";


const FACE_POINTS: number[] = [1, 33, 263, 61, 291, 199]; // Nose, eyes, mouth corners, chin

// 3D model points (from MediaPipe canonical face model)
const MODEL_POINTS: number[] = [
    0, -1.126865, 7.475604,     // Nose tip
    -4.445859, 2.663991, 3.173422,  // Left eye
    4.445859, 2.663991, 3.173422,   // Right eye
    -2.456206, -4.342621, 4.283884, // Left mouth corner
    2.456206, -4.342621, 4.283884,  // Right mouth corner
    0, -9.403378, 4.264492         // Chin
];

class FaceAnalyzer {
    private faceLandmarker: FaceLandmarker | null;

    constructor() {
        this.faceLandmarker = null;
    }

    async initialize(vision: any): Promise<void> {
        this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "IMAGE",
            numFaces: 1
        });
    }

    async detectHeadPose(canvas: HTMLCanvasElement): Promise<HeadPose | null> {
        if (!this.faceLandmarker) {
            return null;
        }

        const result: FaceLandmarkerResult = this.faceLandmarker.detect(canvas);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            return this.calculateHeadPose(result.faceLandmarks[0], canvas);
        }
        return null;
    }

    private calculateHeadPose(landmarks: any, canvasElement: HTMLCanvasElement): HeadPose | null {
        const width = canvasElement.width;
        const height = canvasElement.height;

        // Camera matrix parameters
        const normalizedFocaleY = 1.28; // Typical value for webcams
        const focalLength = height * normalizedFocaleY;
        const centerX = width / 2;
        const centerY = height / 2;

        // Create camera matrix
        const camMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
            focalLength, 0, centerX,
            0, focalLength, centerY,
            0, 0, 1
        ]);

        // Distortion coefficients
        const distCoeffs = cv.matFromArray(4, 1, cv.CV_64FC1, [
            0.1318020374,  // k1
            -0.1550007612, // k2
            -0.0071350401, // p1
            -0.0096747708  // p2
        ]);

        let imagePoints: number[] = [];
        for (const point of FACE_POINTS) {
            imagePoints.push(landmarks[point].x * width);
            imagePoints.push(landmarks[point].y * height);
        }

        if (imagePoints.length > 0) {
            // Create OpenCV matrices
            const imagePointsMat = cv.matFromArray(FACE_POINTS.length, 2, cv.CV_64FC1, imagePoints);
            const modelPointsMat = cv.matFromArray(FACE_POINTS.length, 3, cv.CV_64FC1, MODEL_POINTS);

            // Rotation and translation vectors
            const rvec = new cv.Mat();
            const tvec = new cv.Mat();

            // Solve for pose
            cv.solvePnP(
                modelPointsMat,
                imagePointsMat,
                camMatrix,
                distCoeffs,
                rvec,
                tvec,
                false,
                cv.SOLVEPNP_ITERATIVE
            );

            // Convert rotation vector to rotation matrix
            const rotationMatrix = new cv.Mat();
            const jacobian = new cv.Mat();
            cv.Rodrigues(rvec, rotationMatrix, jacobian);

            // Calculate Euler angles
            const sy = Math.sqrt(
                rotationMatrix.data64F[0] * rotationMatrix.data64F[0] +
                rotationMatrix.data64F[3] * rotationMatrix.data64F[3]
            );

            let x, y, z;
            if (sy > 1e-6) {
                x = Math.atan2(rotationMatrix.data64F[7], rotationMatrix.data64F[8]);
                y = Math.atan2(-rotationMatrix.data64F[6], sy);
                z = Math.atan2(rotationMatrix.data64F[3], rotationMatrix.data64F[0]);
            } else {
                x = Math.atan2(-rotationMatrix.data64F[5], rotationMatrix.data64F[4]);
                y = Math.atan2(-rotationMatrix.data64F[6], sy);
                z = 0;
            }

            // Convert to degrees
            const pitch = (x * 180.0) / Math.PI;
            const yaw = (y * 180.0) / Math.PI;
            const roll = (z * 180.0) / Math.PI;

            // Classify head pose
            let headPose: string;
            if (yaw < -20) {
                headPose = 'LEFT';
            } else if (yaw > 20) {
                headPose = 'RIGHT';
            } else if (pitch > 0 && Math.abs(pitch) < 165) {
                headPose = 'UP';
            } else if (pitch < 0 && Math.abs(pitch) < 165) {
                headPose = 'DOWN';
            } else {
                headPose = 'FORWARD';
            }

            

            // Clean up OpenCV matrices
            imagePointsMat.delete();
            modelPointsMat.delete();
            rvec.delete();
            tvec.delete();
            rotationMatrix.delete();
            jacobian.delete();

            return {
                pitch,
                yaw,
                roll,
                headPose
            };
        }
        return null;
    }
}

interface HeadPose {
    pitch: number;
    yaw: number;
    roll: number;
    headPose: string;
}


class FaceDetectionApp {
    video: HTMLVideoElement;
    anomalyList: HTMLUListElement;
    faceDetector: FaceDetector | null;
    faceAnalyzer: FaceAnalyzer;

    constructor(videoElementId: string, anomalyListId: string) {
        const videoElement = document.getElementById(videoElementId);
        const anomalyElement = document.getElementById(anomalyListId);

        if (!(videoElement instanceof HTMLVideoElement)) {
            throw new Error(`Element with ID ${videoElementId} is not a valid HTMLVideoElement.`);
        }
        if (!(anomalyElement instanceof HTMLUListElement)) {
            throw new Error(`Element with ID ${anomalyListId} is not a valid HTMLUListElement.`);
        }

        this.video = videoElement;
        this.anomalyList = anomalyElement;
        this.faceDetector = null;
        this.faceAnalyzer = new FaceAnalyzer();
    }

    async initialize(): Promise<void> {
        try {
            await this.startWebcam();
            await this.initializeMediaPipes();
            this.startFaceDetection();
        } catch (err) {
            console.error("Initialization error:", err);
        }
    }

    async startWebcam(): Promise<void> {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        this.video.srcObject = stream;
        await this.video.play();
    }

    async initializeMediaPipes(): Promise<void> {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        this.faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            runningMode: "IMAGE"
        });
        await this.faceAnalyzer.initialize(vision);
    }

    startFaceDetection(): void {
        setInterval(() => {
            this.detectFace();
        }, 3000);
    }

    createCanvasElement(videoElement: HTMLVideoElement): HTMLCanvasElement {
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const context = canvas.getContext("2d");
        if (context) {
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            return canvas;
        } else {
            throw new Error("2D context for canvas is null or undefined.");
        }
    }

    async detectFace(): Promise<void> {
        if (!this.faceDetector) return;
        
        const canvas = this.createCanvasElement(this.video);
        let result = await this.faceDetector.detect(canvas);
        const detections = result.detections;

        if (detections.length === 0) {
            this.addAnomaly('No person detected');
        } else if (detections.length > 1) {
            this.addAnomaly('Multiple persons detected');
        } else {
            this.addAnomaly('Person detected');

            if (this.isFaceTurned(detections[0])) {
                const result = this.faceAnalyzer.detectHeadPose(canvas);
                console.log(result);
            }
        }
    }

    isFaceTurned(detection: any): boolean {
        const landmarks = detection.keypoints;
        const imageWidth = detection.boundingBox.width;
        const leftEyeX = landmarks[0].x * imageWidth;
        const rightEyeX = landmarks[1].x * imageWidth;
        const jawLeftX = landmarks[4].x * imageWidth;
        const jawRightX = landmarks[5].x * imageWidth;
        const faceWidth = jawRightX - jawLeftX;
        const faceCenterX = (jawLeftX + jawRightX) / 2;

        const threshold = 0.15;
        return (
            Math.abs(leftEyeX - faceCenterX) < faceWidth * threshold ||
            Math.abs(rightEyeX - faceCenterX) < faceWidth * threshold
        );
    }

    addAnomaly(type: string): void {
        const timestamp = new Date().toLocaleTimeString();
        const li = document.createElement('li');
        li.textContent = `${timestamp}: ${type}`;
        this.anomalyList.appendChild(li);
        this.anomalyList.scrollTop = this.anomalyList.scrollHeight;
    }
}

// Instantiate and initialize the app
const faceDetectionApp = new FaceDetectionApp("webcam", "anomalyList");
faceDetectionApp.initialize();