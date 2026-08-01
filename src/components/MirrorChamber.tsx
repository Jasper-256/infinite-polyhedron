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
uniform vec2 uRotation;
uniform float uZoom;
uniform int uBounces;
uniform vec4 uPlanes[20];
uniform vec3 uEdgeA[30];
uniform vec3 uEdgeB[30];
uniform vec3 uFaceA[20];
uniform vec3 uFaceB[20];
uniform vec3 uFaceC[20];

#define FACE_COUNT 20
#define EDGE_COUNT 30
#define MAX_BOUNCES 24
#define FAR 100.0

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

mat3 rotateX(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat3(
    1.0, 0.0, 0.0,
    0.0, c, s,
    0.0, -s, c
  );
}

mat3 rotateY(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat3(
    c, 0.0, -s,
    0.0, 1.0, 0.0,
    s, 0.0, c
  );
}

float segmentDistance(vec3 p, vec3 a, vec3 b) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

float segmentSegmentDistance(
  vec3 p1,
  vec3 q1,
  vec3 p2,
  vec3 q2,
  out float firstAlong
) {
  vec3 d1 = q1 - p1;
  vec3 d2 = q2 - p2;
  vec3 r = p1 - p2;
  float a = dot(d1, d1);
  float e = dot(d2, d2);
  float f = dot(d2, r);
  float s;
  float t;

  if (a <= 0.000001 && e <= 0.000001) {
    s = 0.0;
    t = 0.0;
  } else if (a <= 0.000001) {
    s = 0.0;
    t = clamp(f / e, 0.0, 1.0);
  } else {
    float c = dot(d1, r);
    if (e <= 0.000001) {
      t = 0.0;
      s = clamp(-c / a, 0.0, 1.0);
    } else {
      float b = dot(d1, d2);
      float denominator = a * e - b * b;
      s = denominator != 0.0
        ? clamp((b * f - c * e) / denominator, 0.0, 1.0)
        : 0.0;
      t = (b * s + f) / e;

      if (t < 0.0) {
        t = 0.0;
        s = clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }

  firstAlong = s;
  return length(
    (p1 + d1 * s) -
    (p2 + d2 * t)
  );
}

float faceEdgeDistance(vec3 point, int faceIndex) {
  vec3 a = uFaceA[faceIndex];
  vec3 b = uFaceB[faceIndex];
  vec3 c = uFaceC[faceIndex];
  return min(
    segmentDistance(point, a, b),
    min(segmentDistance(point, b, c), segmentDistance(point, c, a))
  );
}

bool intersectIcosahedron(
  vec3 ro,
  vec3 rd,
  out float nearT,
  out float farT,
  out int nearFace,
  out int farFace
) {
  nearT = -FAR;
  farT = FAR;
  nearFace = 0;
  farFace = 0;

  for (int i = 0; i < FACE_COUNT; i++) {
    vec3 normal = uPlanes[i].xyz;
    float originSide = uPlanes[i].w - dot(normal, ro);
    float directionSide = dot(normal, rd);

    if (abs(directionSide) < 0.00001) {
      if (originSide < 0.0) return false;
    } else {
      float distance = originSide / directionSide;
      if (directionSide < 0.0) {
        if (distance > nearT) {
          nearT = distance;
          nearFace = i;
        }
      } else {
        if (distance < farT) {
          farT = distance;
          farFace = i;
        }
      }
    }
  }

  return nearT <= farT && farT > 0.0;
}

float intersectInterior(
  vec3 ro,
  vec3 rd,
  out vec3 normal,
  out int faceIndex
) {
  float nearest = FAR;
  normal = vec3(0.0, 0.0, 1.0);
  faceIndex = 0;

  for (int i = 0; i < FACE_COUNT; i++) {
    vec3 faceNormal = uPlanes[i].xyz;
    float denominator = dot(faceNormal, rd);
    if (denominator > 0.00001) {
      float distance =
        (uPlanes[i].w - dot(faceNormal, ro)) / denominator;
      if (distance > 0.0002 && distance < nearest) {
        nearest = distance;
        normal = faceNormal;
        faceIndex = i;
      }
    }
  }
  return nearest;
}

vec3 studioEnvironment(vec3 direction) {
  direction = normalize(direction);
  vec3 low = vec3(0.004, 0.0045, 0.005);
  vec3 high = vec3(0.030, 0.033, 0.036);
  vec3 color = mix(low, high, smoothstep(-0.65, 0.9, direction.y));

  vec3 largeBox = normalize(vec3(-0.62, 0.68, 0.52));
  float boxGlow = pow(max(dot(direction, largeBox), 0.0), 24.0);
  float boxCore = pow(max(dot(direction, largeBox), 0.0), 110.0);
  color += vec3(0.70, 0.73, 0.75) * boxGlow * 0.19;
  color += vec3(1.0, 0.96, 0.90) * boxCore * 1.15;

  vec3 rimBox = normalize(vec3(0.78, 0.15, -0.58));
  float rim = pow(max(dot(direction, rimBox), 0.0), 70.0);
  color += vec3(0.34, 0.42, 0.48) * rim * 0.55;

  float horizon = exp(-abs(direction.y + 0.08) * 30.0);
  color += vec3(0.016, 0.018, 0.019) * horizon;
  return color;
}

vec3 background(vec3 ro, vec3 rd) {
  vec3 color = studioEnvironment(rd) * 0.38;

  if (rd.y < -0.0001) {
    float floorT = (-1.73 - ro.y) / rd.y;
    if (floorT > 0.0) {
      vec3 point = ro + rd * floorT;
      float contact = exp(
        -point.x * point.x * 0.95 -
        point.z * point.z * 0.36
      );
      float broadShadow = exp(
        -point.x * point.x * 0.24 -
        point.z * point.z * 0.11
      );
      vec3 floorReflection = studioEnvironment(
        reflect(rd, vec3(0.0, 1.0, 0.0))
      );
      color = vec3(0.010, 0.0105, 0.011);
      color += floorReflection * 0.055;
      color *= 1.0 - contact * 0.82 - broadShadow * 0.12;
    }
  }

  return color;
}

vec3 traceMirroredInterior(vec3 ro, vec3 rd) {
  const vec3 barColor = vec3(1.0, 0.78, 0.52);
  const vec3 barHot = vec3(1.0, 0.97, 0.88);
  vec3 radiance = vec3(0.0);
  vec3 throughput = vec3(1.0);

  for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
    if (bounce >= uBounces) break;

    vec3 faceNormal;
    int faceIndex;
    float wallT = intersectInterior(ro, rd, faceNormal, faceIndex);
    if (wallT >= FAR - 1.0) break;

    float nearestBar = FAR;
    float nearestAlong = 0.0;
    vec3 rayEnd = ro + rd * wallT;
    for (int edgeIndex = 0; edgeIndex < EDGE_COUNT; edgeIndex++) {
      float rayAlong;
      float distanceToBar = segmentSegmentDistance(
        ro,
        rayEnd,
        uEdgeA[edgeIndex],
        uEdgeB[edgeIndex],
        rayAlong
      );
      if (distanceToBar < nearestBar) {
        nearestBar = distanceToBar;
        nearestAlong = rayAlong * wallT;
      }
    }

    float core = 1.0 -
      smoothstep(0.002, 0.010, nearestBar);
    float tightBloom = exp(-nearestBar * 82.0);
    float softBloom = exp(-nearestBar * 30.0);
    float depthLoss = exp(-float(bounce) * 0.082);
    float airLoss = exp(-nearestAlong * 0.035);
    radiance += throughput * depthLoss * airLoss * (
      barHot * core * 0.72 +
      barColor * tightBloom * 0.038 +
      barColor * softBloom * 0.0012
    );

    if (nearestBar < 0.0035) {
      radiance += throughput * barHot * 0.75;
      break;
    }

    vec3 hit = ro + rd * wallT;
    float edgeDistance = faceEdgeDistance(hit, faceIndex);
    float seam = exp(-edgeDistance * 85.0);
    float barDistance = FAR;
    for (int edgeIndex = 0; edgeIndex < EDGE_COUNT; edgeIndex++) {
      barDistance = min(
        barDistance,
        segmentDistance(
          hit,
          uEdgeA[edgeIndex],
          uEdgeB[edgeIndex]
        )
      );
    }

    float reflectedBarLight = exp(-barDistance * 15.0);
    float faceVariation =
      0.88 + 0.12 * fract(float(faceIndex) * 0.618033);
    float grazing = pow(
      1.0 - abs(dot(faceNormal, -rd)),
      5.0
    );
    float reflectivity = mix(0.82, 0.92, grazing);

    vec3 coating = vec3(0.0065, 0.0072, 0.0078) * faceVariation;
    coating += vec3(0.014, 0.013, 0.012) * seam;
    coating += barColor * reflectedBarLight * 0.014;
    radiance += throughput * coating * (1.0 - reflectivity) * 2.0;

    throughput *= reflectivity;
    throughput *= mix(
      vec3(0.985, 0.965, 0.94),
      vec3(0.96, 0.975, 0.99),
      fract(float(faceIndex) * 0.381966)
    );

    rd = reflect(rd, faceNormal);
    ro = hit - faceNormal * 0.0012;
  }

  return radiance;
}

vec3 acesToneMap(vec3 color) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp(
    (color * (a * color + b)) /
    (color * (c * color + d) + e),
    0.0,
    1.0
  );
}

void main() {
  vec2 screen = vUv * 2.0 - 1.0;
  screen.x *= uResolution.x / uResolution.y;

  vec3 worldRo = vec3(0.0, 0.10, uZoom);
  vec3 worldRd = normalize(vec3(screen * 0.79, -2.18));
  mat3 objectToWorld =
    rotateY(uRotation.x) *
    rotateX(uRotation.y);
  mat3 worldToObject = transpose(objectToWorld);
  vec3 ro = worldToObject * worldRo;
  vec3 rd = normalize(worldToObject * worldRd);

  vec3 color = background(worldRo, worldRd);
  float nearT;
  float farT;
  int nearFace;
  int farFace;

  if (
    intersectIcosahedron(
      ro,
      rd,
      nearT,
      farT,
      nearFace,
      farFace
    ) &&
    nearT > 0.0
  ) {
    vec3 frontNormal = uPlanes[nearFace].xyz;
    vec3 frontHit = ro + rd * nearT;
    vec3 worldNormal = normalize(objectToWorld * frontNormal);
    vec3 reflectedWorld = reflect(worldRd, worldNormal);
    vec3 externalReflection = studioEnvironment(reflectedWorld);

    float facing = clamp(dot(-rd, frontNormal), 0.0, 1.0);
    float fresnel = 0.045 +
      (1.0 - 0.045) * pow(1.0 - facing, 5.0);
    vec3 refracted = refract(rd, frontNormal, 1.0 / 1.47);
    if (length(refracted) < 0.1) refracted = rd;

    vec3 insideOrigin = frontHit - frontNormal * 0.002;
    vec3 interior = traceMirroredInterior(
      insideOrigin,
      normalize(refracted)
    );

    float glassDistance = max(farT - nearT, 0.0);
    vec3 absorption = exp(
      -vec3(0.055, 0.037, 0.020) * glassDistance
    );
    float coatingReflection = 0.15 + fresnel * 0.80;
    float transmission = (1.0 - fresnel) * 0.78;
    color =
      interior * absorption * transmission +
      externalReflection * coatingReflection;

    float edgeDistance = faceEdgeDistance(frontHit, nearFace);
    float hardEdge = 1.0 - smoothstep(0.006, 0.026, edgeDistance);
    float softEdge = exp(-edgeDistance * 28.0);
    vec3 edgeMetal =
      vec3(0.010, 0.0105, 0.011) +
      externalReflection * 0.36;
    color = mix(color, edgeMetal, hardEdge * 0.94);
    color += externalReflection * softEdge * 0.055;

    float silhouette = pow(1.0 - facing, 3.0);
    color += externalReflection * silhouette * 0.48;
    color += vec3(0.018, 0.020, 0.021) * (1.0 - facing) * 0.34;
  }

  float vignette = dot(vUv - 0.5, vUv - 0.5);
  color *= 1.0 - vignette * 0.56;
  float grain =
    hash21(gl_FragCoord.xy + fract(uTime) * 719.31) - 0.5;
  color += grain * 0.0045;
  color = acesToneMap(color * 0.98);
  color = pow(color, vec3(0.4545));
  outColor = vec4(color, 1.0);
}`;

type Point = [number, number, number];

type GeometryData = {
  planes: Float32Array;
  edgeA: Float32Array;
  edgeB: Float32Array;
  faceA: Float32Array;
  faceB: Float32Array;
  faceC: Float32Array;
};

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

  const radius = 1.56;
  const vertices = rawVertices.map(([x, y, z]): Point => {
    const length = Math.hypot(x, y, z);
    return [
      (x / length) * radius,
      (y / length) * radius,
      (z / length) * radius,
    ];
  });

  const distance = (a: Point, b: Point) =>
    Math.hypot(
      a[0] - b[0],
      a[1] - b[1],
      a[2] - b[2],
    );

  let edgeLength = Number.POSITIVE_INFINITY;
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      edgeLength = Math.min(
        edgeLength,
        distance(vertices[i], vertices[j]),
      );
    }
  }

  const faces: [number, number, number][] = [];
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      for (let k = j + 1; k < vertices.length; k++) {
        const isFace =
          Math.abs(
            distance(vertices[i], vertices[j]) - edgeLength,
          ) < 0.001 &&
          Math.abs(
            distance(vertices[j], vertices[k]) - edgeLength,
          ) < 0.001 &&
          Math.abs(
            distance(vertices[k], vertices[i]) - edgeLength,
          ) < 0.001;
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
    const ab: Point = [
      b[0] - a[0],
      b[1] - a[1],
      b[2] - a[2],
    ];
    const ac: Point = [
      c[0] - a[0],
      c[1] - a[1],
      c[2] - a[2],
    ];
    let normal: Point = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const normalLength = Math.hypot(...normal);
    normal = normal.map(
      (value) => value / normalLength,
    ) as Point;

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
      normal[0] * a[0] +
      normal[1] * a[1] +
      normal[2] * a[2];
    planes.push(...normal, planeDistance);
    faceA.push(...a);
    faceB.push(...b);
    faceC.push(...c);
  }

  const edgeA: number[] = [];
  const edgeB: number[] = [];
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      if (
        Math.abs(distance(vertices[i], vertices[j]) - edgeLength) <
        0.001
      ) {
        const a = vertices[i].map((value) => value * 0.91) as Point;
        const b = vertices[j].map((value) => value * 0.91) as Point;
        const trim = 0.035;
        edgeA.push(
          a[0] + (b[0] - a[0]) * trim,
          a[1] + (b[1] - a[1]) * trim,
          a[2] + (b[2] - a[2]) * trim,
        );
        edgeB.push(
          b[0] + (a[0] - b[0]) * trim,
          b[1] + (a[1] - b[1]) * trim,
          b[2] + (a[2] - b[2]) * trim,
        );
      }
    }
  }

  return {
    planes: new Float32Array(planes),
    edgeA: new Float32Array(edgeA),
    edgeB: new Float32Array(edgeB),
    faceA: new Float32Array(faceA),
    faceB: new Float32Array(faceB),
    faceC: new Float32Array(faceC),
  };
}

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Unable to create shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message =
      gl.getShaderInfoLog(shader) ?? "Unknown shader error.";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

export default function MirrorChamber() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState("");
  const controlsRef = useRef({
    dragging: false,
    x: 0,
    y: 0,
    yaw: 0.54,
    pitch: -0.16,
    targetYaw: 0.54,
    targetPitch: -0.16,
    zoom: 5.15,
    targetZoom: 5.15,
    lastInteraction: 0,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      setError("WebGL 2 is required to render this object.");
      return;
    }

    let program: WebGLProgram | null = null;
    let animationFrame = 0;
    let disposed = false;
    const startedAt = performance.now();
    let previousTime = startedAt;

    try {
      const vertexShader = compileShader(
        gl,
        gl.VERTEX_SHADER,
        VERTEX_SHADER,
      );
      const fragmentShader = compileShader(
        gl,
        gl.FRAGMENT_SHADER,
        FRAGMENT_SHADER,
      );
      program = gl.createProgram();
      if (!program) throw new Error("Unable to create WebGL program.");
      const activeProgram = program;
      gl.attachShader(activeProgram, vertexShader);
      gl.attachShader(activeProgram, fragmentShader);
      gl.linkProgram(activeProgram);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);

      if (!gl.getProgramParameter(activeProgram, gl.LINK_STATUS)) {
        throw new Error(
          gl.getProgramInfoLog(activeProgram) ??
            "Unable to link shaders.",
        );
      }

      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 3, -1, -1, 3]),
        gl.STATIC_DRAW,
      );
      const position = gl.getAttribLocation(
        activeProgram,
        "aPosition",
      );
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(
        position,
        2,
        gl.FLOAT,
        false,
        0,
        0,
      );
      gl.useProgram(activeProgram);

      const uniform = (name: string) =>
        gl.getUniformLocation(activeProgram, name);
      const uniforms = {
        resolution: uniform("uResolution"),
        time: uniform("uTime"),
        rotation: uniform("uRotation"),
        zoom: uniform("uZoom"),
        bounces: uniform("uBounces"),
        planes: uniform("uPlanes[0]"),
        edgeA: uniform("uEdgeA[0]"),
        edgeB: uniform("uEdgeB[0]"),
        faceA: uniform("uFaceA[0]"),
        faceB: uniform("uFaceB[0]"),
        faceC: uniform("uFaceC[0]"),
      };

      const geometry = buildIcosahedron();
      gl.uniform4fv(uniforms.planes, geometry.planes);
      gl.uniform3fv(uniforms.edgeA, geometry.edgeA);
      gl.uniform3fv(uniforms.edgeB, geometry.edgeB);
      gl.uniform3fv(uniforms.faceA, geometry.faceA);
      gl.uniform3fv(uniforms.faceB, geometry.faceB);
      gl.uniform3fv(uniforms.faceC, geometry.faceC);

      const isCompact = window.matchMedia(
        "(max-width: 700px)",
      ).matches;
      const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      if (isCompact) {
        controlsRef.current.zoom = 8.5;
        controlsRef.current.targetZoom = 8.5;
      }
      gl.uniform1i(uniforms.bounces, isCompact ? 16 : 22);

      const resize = () => {
        const pixelRatio = Math.min(
          window.devicePixelRatio || 1,
          isCompact ? 1.15 : 1.5,
        );
        const width = Math.max(
          1,
          Math.round(canvas.clientWidth * pixelRatio),
        );
        const height = Math.max(
          1,
          Math.round(canvas.clientHeight * pixelRatio),
        );
        if (
          canvas.width !== width ||
          canvas.height !== height
        ) {
          canvas.width = width;
          canvas.height = height;
          gl.viewport(0, 0, width, height);
        }
      };

      const render = (now: number) => {
        if (disposed || !program) return;
        resize();
        const controls = controlsRef.current;
        const delta = Math.min(
          (now - previousTime) / 1000,
          0.05,
        );
        previousTime = now;

        if (
          !reducedMotion &&
          !controls.dragging &&
          now - controls.lastInteraction > 1800
        ) {
          controls.targetYaw += delta * 0.065;
        }
        controls.yaw +=
          (controls.targetYaw - controls.yaw) * 0.07;
        controls.pitch +=
          (controls.targetPitch - controls.pitch) * 0.07;
        controls.zoom +=
          (controls.targetZoom - controls.zoom) * 0.08;

        gl.useProgram(activeProgram);
        gl.uniform2f(
          uniforms.resolution,
          canvas.width,
          canvas.height,
        );
        gl.uniform1f(
          uniforms.time,
          (now - startedAt) / 1000,
        );
        gl.uniform2f(
          uniforms.rotation,
          controls.yaw,
          controls.pitch,
        );
        gl.uniform1f(uniforms.zoom, controls.zoom);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        animationFrame = window.requestAnimationFrame(render);
      };

      const pointerDown = (event: PointerEvent) => {
        const controls = controlsRef.current;
        controls.dragging = true;
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add("is-dragging");
      };

      const pointerMove = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (!controls.dragging) return;
        controls.targetYaw +=
          (event.clientX - controls.x) * 0.005;
        controls.targetPitch = Math.max(
          -1.15,
          Math.min(
            1.15,
            controls.targetPitch +
              (event.clientY - controls.y) * 0.005,
          ),
        );
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
      };

      const pointerUp = (event: PointerEvent) => {
        const controls = controlsRef.current;
        controls.dragging = false;
        controls.lastInteraction = performance.now();
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        canvas.classList.remove("is-dragging");
      };

      const wheel = (event: WheelEvent) => {
        event.preventDefault();
        const controls = controlsRef.current;
        controls.targetZoom = Math.max(
          4.0,
          Math.min(
            isCompact ? 9.0 : 7.2,
            controls.targetZoom + event.deltaY * 0.0025,
          ),
        );
        controls.lastInteraction = performance.now();
      };

      canvas.addEventListener("pointerdown", pointerDown);
      canvas.addEventListener("pointermove", pointerMove);
      canvas.addEventListener("pointerup", pointerUp);
      canvas.addEventListener("pointercancel", pointerUp);
      canvas.addEventListener("wheel", wheel, {
        passive: false,
      });
      animationFrame = window.requestAnimationFrame(render);

      return () => {
        disposed = true;
        window.cancelAnimationFrame(animationFrame);
        canvas.removeEventListener("pointerdown", pointerDown);
        canvas.removeEventListener("pointermove", pointerMove);
        canvas.removeEventListener("pointerup", pointerUp);
        canvas.removeEventListener(
          "pointercancel",
          pointerUp,
        );
        canvas.removeEventListener("wheel", wheel);
        gl.deleteProgram(activeProgram);
      };
    } catch (caught) {
      const message =
        caught instanceof Error
          ? caught.message
          : "The renderer could not start.";
      setError(message);
      return () => {
        disposed = true;
        window.cancelAnimationFrame(animationFrame);
        if (program) gl.deleteProgram(program);
      };
    }
  }, []);

  return (
    <main className="experience">
      <canvas
        ref={canvasRef}
        className="chamber"
        aria-label="A photorealistic interactive icosahedron with one-way mirrored faces and light bars along its interior edges"
      />
      {error ? (
        <div className="error-panel" role="alert">
          {error}
        </div>
      ) : null}
    </main>
  );
}
