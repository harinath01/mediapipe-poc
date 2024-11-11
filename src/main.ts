import {
  FaceDetector,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

class FaceDetectionApp {
    video: HTMLVideoElement;
    anomalyList: HTMLUListElement;
    faceDetector: FaceDetector | null;

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
        const result = await this.faceDetector.detect(canvas);
        const detections = result.detections;

        if (detections.length === 0) {
            this.addAnomaly('No person detected');
        } else if (detections.length > 1) {
            this.addAnomaly('Multiple persons detected');
        } else {
            this.addAnomaly('Person detected');

            if (this.isFaceTurned(detections[0])) {
                this.addAnomaly('Face turned');
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

