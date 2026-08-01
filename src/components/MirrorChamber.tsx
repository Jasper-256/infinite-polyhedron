"use client";

import { useEffect, useRef, useState } from "react";

const VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

out vec4 outColor;
in vec2 vUv;

uniform vec2 uResolution;
uniform float uTime;
uniform float uYaw;
uniform float uPitch;
uniform float uFov;
uniform int uBounces;
uniform int uPalette;
uniform vec4 uPlanes[20];
uniform vec3 uLights[12];
uniform vec3 uFaceA[20];
uniform vec3 uFaceB[20];
uniform vec3 uFaceC[20];

#define MAX_BOUNCES 28
#define FACE_COUNT 20
#define LIGHT_COUNT 12
#define FAR 100.0

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec3 palettePrimary() {
  if (uPalette == 1) return vec3(1.0, 0.34, 0.08);
  if (uPalette == 2) return vec3(0.76, 0.22, 1.0);
  return vec3(0.12, 0.78, 1.0);
}

vec3 paletteSecondary() {
  if (uPalette == 1) return vec3(1.0, 0.78, 0.24);
  if (uPalette == 2) return vec3(0.20, 0.72, 1.0);
  return vec3(0.78, 0.96, 1.0);
}

float segmentDistance(vec3 p, vec3 a, vec3 b) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

float intersectInterior(
  vec3 ro,
  vec3 rd,
  out vec3 normal,
  out int faceIndex
) {
  float nearest = FAR;
  faceIndex = 0;
  normal = vec3(0.0, 0.0, 1.0);

  for (int i = 0; i < FACE_COUNT; i++) {
    vec3 n = uPlanes[i].xyz;
    float denom = dot(n, rd);
    if (denom > 0.00001) {
      float t = (uPlanes[i].w - dot(n, ro)) / denom;
      if (t > 0.0002 && t < nearest) {
        nearest = t;
        normal = n;
        faceIndex = i;
      }
    }
  }
  return nearest;
}

vec3 acesToneMap(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 traceChamber(vec3 ro, vec3 rd) {
  vec3 primary = palettePrimary();
  vec3 secondary = paletteSecondary();
  vec3 radiance = vec3(0.0);
  vec3 throughput = vec3(1.0);

  for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
    if (bounce >= uBounces) break;

    vec3 normal;
    int faceIndex;
    float wallT = intersectInterior(ro, rd, normal, faceIndex);
    if (wallT >= FAR - 1.0) break;

    float nearestLamp = FAR;
    float lampAlong = 0.0;
    for (int lightIndex = 0; lightIndex < LIGHT_COUNT; lightIndex++) {
      vec3 lamp = uLights[lightIndex];
      float alongRay = clamp(dot(lamp - ro, rd), 0.0, wallT);
      float distanceToRay = length(ro + rd * alongRay - lamp);
      if (distanceToRay < nearestLamp) {
        nearestLamp = distanceToRay;
        lampAlong = alongRay;
      }
    }

    float fineHalo = exp(-nearestLamp * 34.0);
    float broadHalo = exp(-nearestLamp * 9.0);
    float depthFade = exp(-float(bounce) * 0.078);
    float airFade = exp(-lampAlong * 0.055);
    radiance += throughput * depthFade * airFade *
      (primary * fineHalo * 2.7 + secondary * broadHalo * 0.085);

    vec3 hit = ro + rd * wallT;
    vec3 fa = uFaceA[faceIndex];
    vec3 fb = uFaceB[faceIndex];
    vec3 fc = uFaceC[faceIndex];
    float edgeDistance = min(
      segmentDistance(hit, fa, fb),
      min(segmentDistance(hit, fb, fc), segmentDistance(hit, fc, fa))
    );

    float seam = exp(-edgeDistance * 38.0);
    float seamAura = exp(-edgeDistance * 9.0);
    float cornerDistance = FAR;
    float directLight = 0.0;

    for (int lightIndex = 0; lightIndex < LIGHT_COUNT; lightIndex++) {
      vec3 toLight = uLights[lightIndex] - hit;
      float lampDistance = length(toLight);
      cornerDistance = min(cornerDistance, lampDistance);
      vec3 lightDirection = toLight / max(lampDistance, 0.001);
      float incidence = max(dot(-normal, lightDirection), 0.0);
      directLight += incidence / (0.16 + lampDistance * lampDistance);
    }

    float cornerGlow = exp(-cornerDistance * 4.8);
    float facing = 0.5 + 0.5 * abs(dot(normal, rd));
    float fresnel = pow(1.0 - abs(dot(normal, -rd)), 5.0);
    float faceTone = 0.82 + 0.18 * sin(float(faceIndex) * 12.37 + float(bounce));

    vec3 mirrorBody = mix(
      vec3(0.006, 0.010, 0.016),
      primary * 0.075,
      facing * faceTone
    );
    vec3 localLight =
      mirrorBody * (0.18 + directLight * 0.46) +
      secondary * seam * 0.105 +
      primary * seamAura * 0.012 +
      secondary * cornerGlow * 0.34;

    float mirrorTransmission = mix(0.89, 0.955, fresnel);
    radiance += throughput * localLight * (1.0 - mirrorTransmission) * 2.1;
    throughput *= mirrorTransmission;
    throughput *= mix(
      vec3(0.92, 0.965, 1.0),
      vec3(1.0, 0.94, 0.90),
      fract(float(faceIndex) * 0.3819)
    );

    rd = reflect(rd, normal);
    ro = hit - normal * 0.0014;
  }

  return radiance;
}

void main() {
  vec2 pixel = vUv * 2.0 - 1.0;
  pixel.x *= uResolution.x / uResolution.y;

  float cp = cos(uPitch);
  vec3 forward = normalize(vec3(
    sin(uYaw) * cp,
    sin(uPitch),
    cos(uYaw) * cp
  ));
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
  vec3 up = normalize(cross(forward, right));

  float lens = mix(1.42, 0.88, uFov);
  vec3 rd = normalize(forward * lens + right * pixel.x + up * pixel.y);
  vec3 ro = vec3(
    sin(uTime * 0.071) * 0.075,
    cos(uTime * 0.057) * 0.052,
    sin(uTime * 0.043) * 0.035
  );

  vec3 color = traceChamber(ro, rd);

  float radial = dot(vUv - 0.5, vUv - 0.5);
  color *= 1.0 - radial * 0.72;
  color += palettePrimary() * 0.0025 * (1.0 - smoothstep(0.0, 0.5, radial));

  float grain = hash21(gl_FragCoord.xy + fract(uTime) * 713.17) - 0.5;
  color += grain * 0.010;
  color = acesToneMap(color * 1.18);
  color = pow(color, vec3(0.4545));

  outColor = vec4(color, 1.0);
}`;

type GeometryData = {
  planes: Float32Array;
  lights: Float32Array;
  faceA: Float32Array;
  faceB: Float32Array;
  faceC: Float32Array;
};

type Point = [number, number, number];

function buildIcosahedron(): GeometryData {
  const phi = (1 + Math.sqrt(5)) / 2;
  const rawVertices: Point[] = [
    [-1, phi, 0],
    [1, phi, 0],
    [-1, -phi, 0],
    [1, -phi, 0],
    [0, -1, phi],
    [0, 1, phi],
    [0, -1, -phi],
    [0, 1, -phi],
    [phi, 0, -1],
    [phi, 0, 1],
    [-phi, 0, -1],
    [-phi, 0, 1],
  ];

  const radius = 2.8;
  const vertices = rawVertices.map(([x, y, z]): Point => {
    const length = Math.hypot(x, y, z);
    return [(x / length) * radius, (y / length) * radius, (z / length) * radius];
  });

  const distance = (a: Point, b: Point) =>
    Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  let edgeLength = Number.POSITIVE_INFINITY;
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      edgeLength = Math.min(edgeLength, distance(vertices[i], vertices[j]));
    }
  }

  const faces: [number, number, number][] = [];
  const tolerance = 0.001;
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      for (let k = j + 1; k < vertices.length; k++) {
        const isFace =
          Math.abs(distance(vertices[i], vertices[j]) - edgeLength) < tolerance &&
          Math.abs(distance(vertices[j], vertices[k]) - edgeLength) < tolerance &&
          Math.abs(distance(vertices[k], vertices[i]) - edgeLength) < tolerance;
        if (isFace) faces.push([i, j, k]);
      }
    }
  }

  const planes: number[] = [];
  const faceA: number[] = [];
  const faceB: number[] = [];
  const faceC: number[] = [];

  for (const [ai, bi, ci] of faces) {
    const a = vertices[ai];
    let b = vertices[bi];
    let c = vertices[ci];
    const ab: Point = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac: Point = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal: Point = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const normalLength = Math.hypot(...normal);
    normal = normal.map((value) => value / normalLength) as Point;

    const centroid: Point = [
      (a[0] + b[0] + c[0]) / 3,
      (a[1] + b[1] + c[1]) / 3,
      (a[2] + b[2] + c[2]) / 3,
    ];
    if (
      normal[0] * centroid[0] +
        normal[1] * centroid[1] +
        normal[2] * centroid[2] <
      0
    ) {
      normal = normal.map((value) => -value) as Point;
      [b, c] = [c, b];
    }

    const planeDistance =
      normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2];
    planes.push(...normal, planeDistance);
    faceA.push(...a);
    faceB.push(...b);
    faceC.push(...c);
  }

  const lights = vertices.flatMap(([x, y, z]) => [x * 0.885, y * 0.885, z * 0.885]);

  return {
    planes: new Float32Array(planes),
    lights: new Float32Array(lights),
    faceA: new Float32Array(faceA),
    faceB: new Float32Array(faceB),
    faceC: new Float32Array(faceC),
  };
}

function makeShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Unable to create shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? "Unknown shader error.";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

export default function MirrorChamber() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const interactionRef = useRef({
    dragging: false,
    x: 0,
    y: 0,
    yaw: 0.41,
    pitch: 0.16,
    targetYaw: 0.41,
    targetPitch: 0.16,
    fov: 0.31,
    targetFov: 0.31,
    lastInteraction: 0,
  });
  const pausedRef = useRef(false);
  const paletteRef = useRef(0);
  const [paused, setPaused] = useState(false);
  const [palette, setPalette] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setError("This optical study needs a browser with WebGL 2.");
      return;
    }

    let program: WebGLProgram | null = null;
    let frame = 0;
    let disposed = false;
    const startedAt = performance.now();
    let previousTime = startedAt;

    try {
      const vertexShader = makeShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
      const fragmentShader = makeShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
      program = gl.createProgram();
      if (!program) throw new Error("Unable to create WebGL program.");
      const activeProgram = program;
      gl.attachShader(activeProgram, vertexShader);
      gl.attachShader(activeProgram, fragmentShader);
      gl.linkProgram(activeProgram);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      if (!gl.getProgramParameter(activeProgram, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(activeProgram) ?? "Unable to link shaders.");
      }

      const vertices = new Float32Array([-1, -1, 3, -1, -1, 3]);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

      const position = gl.getAttribLocation(activeProgram, "aPosition");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gl.useProgram(activeProgram);

      const uniform = (name: string) => gl.getUniformLocation(activeProgram, name);
      const uniforms = {
        resolution: uniform("uResolution"),
        time: uniform("uTime"),
        yaw: uniform("uYaw"),
        pitch: uniform("uPitch"),
        fov: uniform("uFov"),
        bounces: uniform("uBounces"),
        palette: uniform("uPalette"),
        planes: uniform("uPlanes[0]"),
        lights: uniform("uLights[0]"),
        faceA: uniform("uFaceA[0]"),
        faceB: uniform("uFaceB[0]"),
        faceC: uniform("uFaceC[0]"),
      };

      const geometry = buildIcosahedron();
      gl.uniform4fv(uniforms.planes, geometry.planes);
      gl.uniform3fv(uniforms.lights, geometry.lights);
      gl.uniform3fv(uniforms.faceA, geometry.faceA);
      gl.uniform3fv(uniforms.faceB, geometry.faceB);
      gl.uniform3fv(uniforms.faceC, geometry.faceC);

      const compactScreen = window.matchMedia("(max-width: 700px)").matches;
      gl.uniform1i(uniforms.bounces, compactScreen ? 18 : 26);

      const resize = () => {
        const pixelRatio = Math.min(window.devicePixelRatio || 1, compactScreen ? 1.15 : 1.5);
        const width = Math.max(1, Math.round(canvas.clientWidth * pixelRatio));
        const height = Math.max(1, Math.round(canvas.clientHeight * pixelRatio));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
          gl.viewport(0, 0, width, height);
        }
      };

      const render = (now: number) => {
        if (disposed || !program) return;
        resize();
        const controls = interactionRef.current;
        const delta = Math.min((now - previousTime) / 1000, 0.05);
        previousTime = now;

        if (!pausedRef.current && !controls.dragging && now - controls.lastInteraction > 2400) {
          controls.targetYaw += delta * 0.028;
          controls.targetPitch = 0.15 + Math.sin(now * 0.00011) * 0.075;
        }
        controls.yaw += (controls.targetYaw - controls.yaw) * 0.065;
        controls.pitch += (controls.targetPitch - controls.pitch) * 0.065;
        controls.fov += (controls.targetFov - controls.fov) * 0.07;

        gl.useProgram(program);
        gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
        gl.uniform1f(uniforms.time, (now - startedAt) / 1000);
        gl.uniform1f(uniforms.yaw, controls.yaw);
        gl.uniform1f(uniforms.pitch, controls.pitch);
        gl.uniform1f(uniforms.fov, controls.fov);
        gl.uniform1i(uniforms.palette, paletteRef.current);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        frame = window.requestAnimationFrame(render);
      };

      const pointerDown = (event: PointerEvent) => {
        const controls = interactionRef.current;
        controls.dragging = true;
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add("is-dragging");
      };
      const pointerMove = (event: PointerEvent) => {
        const controls = interactionRef.current;
        if (!controls.dragging) return;
        const scale = 0.0042;
        controls.targetYaw -= (event.clientX - controls.x) * scale;
        controls.targetPitch = Math.max(
          -1.12,
          Math.min(1.12, controls.targetPitch + (event.clientY - controls.y) * scale),
        );
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
      };
      const pointerUp = (event: PointerEvent) => {
        const controls = interactionRef.current;
        controls.dragging = false;
        controls.lastInteraction = performance.now();
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        canvas.classList.remove("is-dragging");
      };
      const wheel = (event: WheelEvent) => {
        event.preventDefault();
        const controls = interactionRef.current;
        controls.targetFov = Math.max(
          0.02,
          Math.min(0.92, controls.targetFov + event.deltaY * 0.00055),
        );
        controls.lastInteraction = performance.now();
      };

      canvas.addEventListener("pointerdown", pointerDown);
      canvas.addEventListener("pointermove", pointerMove);
      canvas.addEventListener("pointerup", pointerUp);
      canvas.addEventListener("pointercancel", pointerUp);
      canvas.addEventListener("wheel", wheel, { passive: false });
      setReady(true);
      frame = window.requestAnimationFrame(render);

      return () => {
        disposed = true;
        window.cancelAnimationFrame(frame);
        canvas.removeEventListener("pointerdown", pointerDown);
        canvas.removeEventListener("pointermove", pointerMove);
        canvas.removeEventListener("pointerup", pointerUp);
        canvas.removeEventListener("pointercancel", pointerUp);
        canvas.removeEventListener("wheel", wheel);
        if (program) gl.deleteProgram(program);
      };
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "The renderer could not start.";
      setError(message);
      return () => {
        disposed = true;
        window.cancelAnimationFrame(frame);
        if (program) gl.deleteProgram(program);
      };
    }
  }, []);

  const togglePaused = () => {
    const next = !pausedRef.current;
    pausedRef.current = next;
    setPaused(next);
    interactionRef.current.lastInteraction = performance.now();
  };

  const cyclePalette = () => {
    const next = (paletteRef.current + 1) % 3;
    paletteRef.current = next;
    setPalette(next);
    interactionRef.current.lastInteraction = performance.now();
  };

  const paletteNames = ["Cyan", "Solar", "Ultraviolet"];

  return (
    <main className={`experience palette-${palette}`}>
      <canvas
        ref={canvasRef}
        className="chamber"
        aria-label="An interactive ray-traced view from inside a mirrored icosahedron"
      />

      <div className="interface">
        <header className="masthead">
          <div className="brand" aria-label="Infinite Icosahedron">
            <span className="eyebrow">Optical study / 01</span>
            <h1>
              Infinite
              <br />
              Icosahedron
            </h1>
          </div>
          <div className="specimen" aria-hidden="true">
            <span>20 faces</span>
            <span className="specimen-mark">✦</span>
            <span>12 lights</span>
          </div>
        </header>

        <div className="depth-tag">
          <span className="depth-dot" />
          <span>{typeof window !== "undefined" && window.innerWidth <= 700 ? "18" : "26"} mirror bounces</span>
        </div>

        <footer className="controls">
          <p className="gesture">
            <span className="gesture-desktop">Drag to orbit · Scroll to shift perspective</span>
            <span className="gesture-mobile">Drag to look around</span>
          </p>
          <div className="control-group">
            <button type="button" onClick={cyclePalette} className="text-control">
              <span className="control-label">Spectrum</span>
              <span>{paletteNames[palette]}</span>
            </button>
            <button
              type="button"
              onClick={togglePaused}
              className="icon-control"
              aria-label={paused ? "Resume rotation" : "Pause rotation"}
              aria-pressed={paused}
            >
              {paused ? (
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M5.25 3.2 12 8l-6.75 4.8V3.2Z" />
                </svg>
              ) : (
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M4.75 3.25h2v9.5h-2zM9.25 3.25h2v9.5h-2z" />
                </svg>
              )}
            </button>
          </div>
        </footer>
      </div>

      <div className={`loading ${ready || error ? "loading-hidden" : ""}`}>
        <span className="loading-mark" />
        <span>Aligning mirrors</span>
      </div>

      {error ? (
        <div className="error-panel" role="alert">
          <span>Renderer unavailable</span>
          <p>{error}</p>
        </div>
      ) : null}

      <div className="noise" aria-hidden="true" />
    </main>
  );
}
