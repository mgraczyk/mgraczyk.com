import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const ui = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#overlay"),
  stage: document.querySelector("#stage"),
  startPanel: document.querySelector("#startPanel"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  statusText: document.querySelector("#statusText"),
  trackingHint: document.querySelector("#trackingHint"),
  effectPicker: document.querySelector("#effectPicker"),
  fpsBadge: document.querySelector("#fpsBadge"),
};

let faceLandmarker = null;
let mediaStream = null;
let animationId = 0;
let currentEffect = 0;
let running = false;
let lastVideoTime = -1;
let lastDetectAt = 0;
let faceLastSeenAt = 0;
let fpsWindowStart = performance.now();
let fpsFrames = 0;
let smoothedEyes = null;

const gl = ui.canvas.getContext("webgl", {
  alpha: true,
  antialias: true,
  premultipliedAlpha: true,
  powerPreference: "high-performance",
});

if (!gl) {
  ui.statusText.textContent = "WebGL is not available in this browser.";
  ui.startButton.disabled = true;
  throw new Error("WebGL unavailable");
}

const vertexShaderSource = `
attribute vec2 a_position;
uniform vec2 u_center;
uniform vec2 u_size;
uniform float u_angle;
varying vec2 v_uv;

void main() {
  float c = cos(u_angle);
  float s = sin(u_angle);
  vec2 p = a_position * u_size;
  p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
  gl_Position = vec4(u_center + p, 0.0, 1.0);
  v_uv = a_position * 0.5 + 0.5;
}
`;

const fragmentShaderSource = `
precision mediump float;
varying vec2 v_uv;
uniform float u_time;
uniform float u_effect;
uniform float u_eye_side;

const float PI = 3.14159265359;

float ellipse(vec2 p, vec2 r) {
  return length(p / r);
}

float starMask(vec2 p, float points, float innerR, float outerR) {
  float a = atan(p.y, p.x);
  float r = length(p);
  float sector = PI / points;
  float folded = abs(mod(a + sector, 2.0 * sector) - sector);
  float edge = mix(outerR, innerR, smoothstep(0.0, sector, folded));
  return 1.0 - smoothstep(edge - 0.035, edge + 0.035, r);
}

float heartMask(vec2 p) {
  p.y += 0.08;
  p.x *= 1.05;
  float x = p.x;
  float y = p.y;
  float a = x*x + y*y - 0.58;
  float f = a*a*a - x*x*y*y*y;
  return 1.0 - smoothstep(-0.03, 0.03, f);
}

void main() {
  vec2 p = (v_uv - 0.5) * 2.0;
  vec4 color = vec4(0.0);
  float aa = 0.035;

  if (u_effect < 0.5) {
    float shell = 1.0 - smoothstep(0.95, 1.0, ellipse(p, vec2(0.86, 0.96)));
    float border = 1.0 - smoothstep(0.90, 0.98, ellipse(p, vec2(0.86, 0.96)));
    vec2 wobble = vec2(
      sin(u_time * 2.0 + u_eye_side * 1.7),
      cos(u_time * 1.55 + u_eye_side * 2.9)
    ) * 0.17;
    float pupil = 1.0 - smoothstep(0.26, 0.31, length(p - wobble));
    color = mix(vec4(0.04,0.04,0.05,1.0), vec4(1.0), border);
    color = mix(color, vec4(0.02,0.02,0.025,1.0), pupil);
    color.a *= shell;
  } else if (u_effect < 1.5) {
    vec2 hp = vec2(p.x * 0.92, -p.y * 0.98);
    float heart = heartMask(hp);
    float rim = heartMask(hp * 1.10);
    color = mix(vec4(0.18,0.02,0.06,1.0), vec4(1.0,0.16,0.34,1.0), heart);
    color.a = max(heart, rim * 0.75);
  } else if (u_effect < 2.5) {
    float shell = 1.0 - smoothstep(0.95, 1.0, length(p));
    float angle = atan(p.y, p.x);
    float radius = length(p);
    float spiral = 0.5 + 0.5 * sin(angle * 4.0 + radius * 18.0 - u_time * 4.2);
    vec3 ink = mix(vec3(0.04,0.04,0.05), vec3(0.98), smoothstep(0.43,0.57,spiral));
    color = vec4(ink, shell);
  } else if (u_effect < 3.5) {
    float star = starMask(p * 0.92, 5.0, 0.43, 0.98);
    float glow = starMask(p * 1.02, 5.0, 0.43, 0.98);
    vec3 gold = vec3(1.0, 0.72 + 0.12*sin(u_time*2.0), 0.15);
    color = vec4(gold, max(star, glow * 0.55));
  } else if (u_effect < 4.5) {
    float shell = 1.0 - smoothstep(0.96, 1.0, ellipse(p, vec2(0.72, 1.0)));
    float inner = 1.0 - smoothstep(0.86, 0.92, ellipse(p, vec2(0.72, 1.0)));
    float slit = 1.0 - smoothstep(0.07, 0.11, abs(p.x + 0.035*sin(u_time*1.4)));
    slit *= 1.0 - smoothstep(0.63, 0.82, abs(p.y));
    vec3 alien = vec3(0.48, 1.0, 0.43);
    color = mix(vec4(0.04,0.08,0.04,1.0), vec4(alien,1.0), inner);
    color = mix(color, vec4(0.02,0.02,0.025,1.0), slit);
    color.a *= shell;
  } else {
    float shell = 1.0 - smoothstep(0.96, 1.0, length(p));
    float d1 = abs(p.y - p.x);
    float d2 = abs(p.y + p.x);
    float xmark = 1.0 - smoothstep(0.12, 0.18, min(d1, d2));
    xmark *= 1.0 - smoothstep(0.64, 0.88, length(p));
    color = mix(vec4(0.97,0.97,0.97,1.0), vec4(0.03,0.03,0.04,1.0), xmark);
    color.a *= shell;
  }

  if (color.a < 0.01) discard;
  gl_FragColor = color;
}
`;

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info}`);
  }
  return shader;
}

function createProgram() {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

const program = createProgram();
const positionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
]), gl.STATIC_DRAW);

const locations = {
  position: gl.getAttribLocation(program, "a_position"),
  center: gl.getUniformLocation(program, "u_center"),
  size: gl.getUniformLocation(program, "u_size"),
  angle: gl.getUniformLocation(program, "u_angle"),
  time: gl.getUniformLocation(program, "u_time"),
  effect: gl.getUniformLocation(program, "u_effect"),
  eyeSide: gl.getUniformLocation(program, "u_eye_side"),
};

gl.useProgram(program);
gl.enableVertexAttribArray(locations.position);
gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

function resizeCanvas() {
  const rect = ui.stage.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
}

const resizeObserver = new ResizeObserver(resizeCanvas);
resizeObserver.observe(ui.stage);
resizeCanvas();

function setStatus(message, isError = false) {
  ui.statusText.textContent = message;
  ui.statusText.style.color = isError ? "var(--danger)" : "";
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = {
    baseOptions: {
      modelAssetPath: FACE_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  };

  try {
    return await FaceLandmarker.createFromOptions(vision, options);
  } catch (gpuError) {
    console.warn("MediaPipe GPU delegate failed; retrying without an explicit delegate.", gpuError);
    delete options.baseOptions.delegate;
    return await FaceLandmarker.createFromOptions(vision, options);
  }
}

async function init() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not expose camera access.", true);
    return;
  }
  if (!window.isSecureContext && location.hostname !== "localhost") {
    setStatus("Camera access needs HTTPS (or localhost).", true);
    return;
  }

  try {
    faceLandmarker = await createLandmarker();
    setStatus("Tracker ready. Camera access stays under your control.");
    ui.startButton.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("Could not load the face tracker. Check your connection and reload.", true);
  }
}

async function startCamera() {
  if (!faceLandmarker || running) return;
  ui.startButton.disabled = true;
  setStatus("Requesting camera permission…");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 720 },
        height: { ideal: 960 },
        frameRate: { ideal: 30, max: 60 },
      },
    });

    ui.video.srcObject = mediaStream;
    await ui.video.play();
    running = true;
    lastVideoTime = -1;
    lastDetectAt = 0;
    faceLastSeenAt = performance.now();
    smoothedEyes = null;
    ui.startPanel.hidden = true;
    ui.stopButton.disabled = false;
    ui.trackingHint.hidden = false;
    animationId = requestAnimationFrame(frameLoop);
  } catch (error) {
    console.error(error);
    let message = "Could not start the camera.";
    if (error?.name === "NotAllowedError") message = "Camera permission was denied. Allow camera access and try again.";
    if (error?.name === "NotFoundError") message = "No front-facing camera was found.";
    if (error?.name === "NotReadableError") message = "The camera is already in use by another app or tab.";
    setStatus(message, true);
    ui.startButton.disabled = false;
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(animationId);
  animationId = 0;
  mediaStream?.getTracks().forEach(track => track.stop());
  mediaStream = null;
  ui.video.srcObject = null;
  smoothedEyes = null;
  clearCanvas();
  ui.trackingHint.hidden = true;
  ui.stopButton.disabled = true;
  ui.startPanel.hidden = false;
  setStatus("Tracker ready. Camera is off.");
  ui.startButton.disabled = false;
  ui.fpsBadge.textContent = "-- fps";
}

function clearCanvas() {
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

function mapLandmarkToStage(landmark) {
  const stageRect = ui.stage.getBoundingClientRect();
  const stageW = stageRect.width;
  const stageH = stageRect.height;
  const videoW = ui.video.videoWidth || 1;
  const videoH = ui.video.videoHeight || 1;

  const coverScale = Math.max(stageW / videoW, stageH / videoH);
  const shownW = videoW * coverScale;
  const shownH = videoH * coverScale;
  const offsetX = (stageW - shownW) * 0.5;
  const offsetY = (stageH - shownH) * 0.5;

  const unmirroredX = offsetX + landmark.x * shownW;
  return {
    x: stageW - unmirroredX,
    y: offsetY + landmark.y * shownH,
  };
}

function eyeFromCorners(landmarks, aIndex, bIndex, upperIndex, lowerIndex) {
  const a = mapLandmarkToStage(landmarks[aIndex]);
  const b = mapLandmarkToStage(landmarks[bIndex]);
  const upper = mapLandmarkToStage(landmarks[upperIndex]);
  const lower = mapLandmarkToStage(landmarks[lowerIndex]);

  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const cornerDistance = Math.hypot(dx, dy);
  const lidDistance = Math.hypot(lower.x - upper.x, lower.y - upper.y);
  const width = Math.max(30, cornerDistance * 1.42);
  const height = Math.max(width * 0.66, lidDistance * 3.5);

  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
    width,
    height: Math.min(height, width * 0.92),
    angle: Math.atan2(dy, dx),
  };
}

function normalizeAngle(angle) {
  while (angle > Math.PI * 0.5) angle -= Math.PI;
  while (angle < -Math.PI * 0.5) angle += Math.PI;
  return angle;
}

function getEyes(landmarks) {
  const first = eyeFromCorners(landmarks, 33, 133, 159, 145);
  const second = eyeFromCorners(landmarks, 362, 263, 386, 374);
  first.angle = normalizeAngle(first.angle);
  second.angle = normalizeAngle(second.angle);
  return [first, second];
}

function lerpAngle(a, b, t) {
  let delta = b - a;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return a + delta * t;
}

function smoothEyes(nextEyes) {
  if (!smoothedEyes) {
    smoothedEyes = nextEyes.map(eye => ({ ...eye }));
    return smoothedEyes;
  }

  const alpha = 0.48;
  for (let i = 0; i < 2; i += 1) {
    const dst = smoothedEyes[i];
    const src = nextEyes[i];
    dst.x += (src.x - dst.x) * alpha;
    dst.y += (src.y - dst.y) * alpha;
    dst.width += (src.width - dst.width) * alpha;
    dst.height += (src.height - dst.height) * alpha;
    dst.angle = lerpAngle(dst.angle, src.angle, alpha);
  }
  return smoothedEyes;
}

function renderEyes(eyes, now) {
  resizeCanvas();
  clearCanvas();
  const rect = ui.stage.getBoundingClientRect();
  const stageW = rect.width;
  const stageH = rect.height;
  if (!stageW || !stageH) return;

  gl.useProgram(program);
  gl.uniform1f(locations.time, now * 0.001);
  gl.uniform1f(locations.effect, currentEffect);

  eyes.forEach((eye, index) => {
    const centerX = (eye.x / stageW) * 2 - 1;
    const centerY = 1 - (eye.y / stageH) * 2;
    const halfWidthClip = eye.width / stageW;
    const halfHeightClip = eye.height / stageH;

    gl.uniform2f(locations.center, centerX, centerY);
    gl.uniform2f(locations.size, halfWidthClip, halfHeightClip);
    gl.uniform1f(locations.angle, -eye.angle);
    gl.uniform1f(locations.eyeSide, index === 0 ? -1 : 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  });
}

function updateFps(now) {
  fpsFrames += 1;
  const elapsed = now - fpsWindowStart;
  if (elapsed >= 700) {
    const fps = Math.round((fpsFrames * 1000) / elapsed);
    ui.fpsBadge.textContent = `${fps} fps`;
    fpsFrames = 0;
    fpsWindowStart = now;
  }
}

function frameLoop(now) {
  if (!running) return;
  updateFps(now);

  const videoReady = ui.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && ui.video.videoWidth > 0;
  if (videoReady && ui.video.currentTime !== lastVideoTime && now - lastDetectAt >= 32) {
    lastVideoTime = ui.video.currentTime;
    lastDetectAt = now;

    try {
      const result = faceLandmarker.detectForVideo(ui.video, now);
      const landmarks = result.faceLandmarks?.[0];
      if (landmarks) {
        faceLastSeenAt = now;
        const eyes = smoothEyes(getEyes(landmarks));
        renderEyes(eyes, now);
        ui.trackingHint.hidden = true;
      }
    } catch (error) {
      console.error("Face tracking frame failed", error);
    }
  }

  if (now - faceLastSeenAt > 350) {
    smoothedEyes = null;
    clearCanvas();
    ui.trackingHint.hidden = false;
  } else if (smoothedEyes) {
    renderEyes(smoothedEyes, now);
  }

  animationId = requestAnimationFrame(frameLoop);
}

ui.startButton.addEventListener("click", startCamera);
ui.stopButton.addEventListener("click", stopCamera);
ui.effectPicker.addEventListener("click", (event) => {
  const button = event.target.closest("[data-effect]");
  if (!button) return;
  currentEffect = Number(button.dataset.effect) || 0;
  ui.effectPicker.querySelectorAll("[data-effect]").forEach(option => {
    const selected = option === button;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-checked", String(selected));
  });
});

window.addEventListener("pagehide", () => {
  if (running) stopCamera();
  faceLandmarker?.close?.();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) clearCanvas();
});

init();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(error => {
      console.warn("Service worker registration failed", error);
    });
  });
}
