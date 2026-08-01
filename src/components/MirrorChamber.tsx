"use client";

import { useEffect, useRef, useState } from "react";

// Rendering quality controls. Keep these near the top so performance can be
// tuned without changing the renderer itself.
const RENDER_PIXEL_RATIO = 2;
const REFLECTIONS_PER_PIXEL = 16;
const POST_PROCESS_TEXTURE_SAMPLES_PER_PIXEL = 20;

const MAX_REFLECTIONS_PER_PIXEL = 16;
const MAX_POST_PROCESS_TEXTURE_SAMPLES_PER_PIXEL = 20;
const SHADER_MAX_REFLECTIONS = 24;
const REFERENCE_FRAME_DURATION_MS = 1000 / 60;
const ROTATION_FOLLOW_PER_REFERENCE_FRAME = 0.07;

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
uniform mat3 uRotation;
uniform float uZoom;
uniform int uBounces;
uniform vec4 uPlanes[20];
uniform vec4 uFaceEdgeOriginA[20];
uniform vec4 uFaceEdgeOriginB[20];
uniform vec4 uFaceEdgeOriginC[20];
uniform vec4 uFaceEdgeDirectionA[20];
uniform vec4 uFaceEdgeDirectionB[20];
uniform vec4 uFaceEdgeDirectionC[20];
uniform vec3 uFrameA[30];
uniform vec3 uFrameB[30];
uniform vec3 uFaceA[20];
uniform vec3 uFaceB[20];
uniform vec3 uFaceC[20];
uniform vec4 uBounceLighting[${SHADER_MAX_REFLECTIONS}];

#define FACE_COUNT 20
#define EDGE_COUNT 30
#define MAX_BOUNCES ${SHADER_MAX_REFLECTIONS}
#define FAR 100.0

const float LIGHT_CORE_RADIUS = 0.014;
const float LIGHT_RAIL_RADIUS = 0.036;
const float MIRROR_EDGE_INSET = 0.043;
const float BOUNDING_RADIUS_SQUARED = 2.5921;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float segmentDistance(vec3 p, vec3 a, vec3 b) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

float segmentSegmentDistanceSquared(
  vec3 p1,
  vec3 q1,
  vec4 edgeOriginData,
  vec4 edgeDirectionData,
  out float firstAlong
) {
  vec3 p2 = edgeOriginData.xyz;
  vec3 d1 = q1 - p1;
  vec3 d2 = edgeDirectionData.xyz;
  vec3 r = p1 - p2;
  float a = dot(d1, d1);
  float e = edgeDirectionData.w;
  float inverseE = edgeOriginData.w;
  float f = dot(d2, r);
  float s;
  float t;

  if (a <= 0.000001) {
    s = 0.0;
    t = clamp(f * inverseE, 0.0, 1.0);
  } else {
    float c = dot(d1, r);
    float b = dot(d1, d2);
    float denominator = a * e - b * b;
    s = denominator != 0.0
      ? clamp((b * f - c * e) / denominator, 0.0, 1.0)
      : 0.0;
    t = (b * s + f) * inverseE;

    if (t < 0.0) {
      t = 0.0;
      s = clamp(-c / a, 0.0, 1.0);
    } else if (t > 1.0) {
      t = 1.0;
      s = clamp((b - c) / a, 0.0, 1.0);
    }
  }

  firstAlong = s;
  vec3 separation =
    (p1 + d1 * s) -
    (p2 + d2 * t);
  return dot(separation, separation);
}

bool intersectsBoundingSphere(vec3 ro, vec3 rd) {
  float towardCenter = dot(ro, rd);
  float originDistanceSquared =
    dot(ro, ro) - BOUNDING_RADIUS_SQUARED;
  float discriminant =
    towardCenter * towardCenter - originDistanceSquared;
  return discriminant >= 0.0 &&
    (towardCenter < 0.0 || originDistanceSquared <= 0.0);
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

float intersectCapsule(
  vec3 ro,
  vec3 rd,
  vec3 pointA,
  vec3 pointB,
  float radius,
  out vec3 hitNormal
) {
  vec3 ba = pointB - pointA;
  vec3 oa = ro - pointA;
  float baba = dot(ba, ba);
  float bard = dot(ba, rd);
  float baoa = dot(ba, oa);
  float rdoa = dot(rd, oa);
  float oaoa = dot(oa, oa);
  float a = baba - bard * bard;
  float b = baba * rdoa - baoa * bard;
  float c =
    baba * oaoa -
    baoa * baoa -
    radius * radius * baba;
  float discriminant = b * b - a * c;

  if (discriminant < 0.0 || abs(a) < 0.000001) return FAR;
  float distance = (-b - sqrt(discriminant)) / a;
  float segmentPosition = baoa + distance * bard;

  if (
    distance > 0.0 &&
    segmentPosition > 0.0 &&
    segmentPosition < baba
  ) {
    vec3 hit = oa + rd * distance -
      ba * segmentPosition / baba;
    hitNormal = normalize(hit);
    return distance;
  }

  vec3 sphereCenter =
    segmentPosition <= 0.0 ? pointA : pointB;
  vec3 sphereOffset = ro - sphereCenter;
  float sphereB = dot(rd, sphereOffset);
  float sphereC =
    dot(sphereOffset, sphereOffset) - radius * radius;
  float sphereDiscriminant =
    sphereB * sphereB - sphereC;
  if (sphereDiscriminant < 0.0) return FAR;
  distance = -sphereB - sqrt(sphereDiscriminant);
  if (distance <= 0.0) return FAR;
  hitNormal = normalize(
    ro + rd * distance - sphereCenter
  );
  return distance;
}

bool intersectExteriorFrame(
  vec3 ro,
  vec3 rd,
  out float nearest,
  out vec3 hitNormal
) {
  nearest = FAR;
  hitNormal = vec3(0.0, 1.0, 0.0);
  for (int edgeIndex = 0; edgeIndex < EDGE_COUNT; edgeIndex++) {
    vec3 normal;
    float distance = intersectCapsule(
      ro,
      rd,
      uFrameA[edgeIndex],
      uFrameB[edgeIndex],
      0.047,
      normal
    );
    if (distance < nearest) {
      nearest = distance;
      hitNormal = normal;
    }
  }
  return nearest < FAR - 1.0;
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
  vec3 wallColor = color;

  if (rd.y < -0.0001) {
    float floorT = (-1.50 - ro.y) / rd.y;
    if (floorT > 0.0) {
      vec3 point = ro + rd * floorT;
      float contact = exp(
        -point.x * point.x * 2.2 -
        point.z * point.z * 1.05
      );
      float broadShadow = exp(
        -point.x * point.x * 0.52 -
        point.z * point.z * 0.24
      );
      vec3 floorReflection = studioEnvironment(
        reflect(rd, vec3(0.0, 1.0, 0.0))
      );
      float concrete = hash21(point.xz * 93.7) - 0.5;
      vec3 floorColor =
        vec3(0.023, 0.0225, 0.0215) + concrete * 0.0016;
      floorColor += floorReflection * 0.060;
      floorColor *= 1.0 - contact * 0.86 - broadShadow * 0.10;
      floorColor += vec3(0.08, 0.13, 0.17) * contact * 0.022;
      float floorBlend = smoothstep(0.005, 0.115, -rd.y);
      color = mix(wallColor, floorColor, floorBlend);
    }
  }

  return color;
}

vec3 traceMirroredInterior(vec3 ro, vec3 rd, int entryFace) {
  vec3 radiance = vec3(0.0);
  vec3 throughput = vec3(1.0);
  vec4 entryEdgeOriginA = uFaceEdgeOriginA[entryFace];
  vec4 entryEdgeOriginB = uFaceEdgeOriginB[entryFace];
  vec4 entryEdgeOriginC = uFaceEdgeOriginC[entryFace];
  vec4 entryEdgeDirectionA = uFaceEdgeDirectionA[entryFace];
  vec4 entryEdgeDirectionB = uFaceEdgeDirectionB[entryFace];
  vec4 entryEdgeDirectionC = uFaceEdgeDirectionC[entryFace];

  for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
    if (bounce >= uBounces) break;

    vec3 faceNormal;
    int faceIndex;
    float wallT = intersectInterior(ro, rd, faceNormal, faceIndex);
    if (wallT >= FAR - 1.0) break;

    float nearestBarSquared = FAR * FAR;
    float nearestAlong = 0.0;
    vec3 rayEnd = ro + rd * wallT;
    vec4 exitEdgeOriginA = uFaceEdgeOriginA[faceIndex];
    vec4 exitEdgeOriginB = uFaceEdgeOriginB[faceIndex];
    vec4 exitEdgeOriginC = uFaceEdgeOriginC[faceIndex];
    vec4 exitEdgeDirectionA = uFaceEdgeDirectionA[faceIndex];
    vec4 exitEdgeDirectionB = uFaceEdgeDirectionB[faceIndex];
    vec4 exitEdgeDirectionC = uFaceEdgeDirectionC[faceIndex];
    for (int candidate = 0; candidate < 6; candidate++) {
      bool useEntryFace = candidate < 3;
      int faceEdgeIndex = candidate - (useEntryFace ? 0 : 3);
      vec4 edgeOrigin;
      vec4 edgeDirection;
      if (faceEdgeIndex == 0) {
        edgeOrigin = useEntryFace
          ? entryEdgeOriginA
          : exitEdgeOriginA;
        edgeDirection = useEntryFace
          ? entryEdgeDirectionA
          : exitEdgeDirectionA;
      } else if (faceEdgeIndex == 1) {
        edgeOrigin = useEntryFace
          ? entryEdgeOriginB
          : exitEdgeOriginB;
        edgeDirection = useEntryFace
          ? entryEdgeDirectionB
          : exitEdgeDirectionB;
      } else {
        edgeOrigin = useEntryFace
          ? entryEdgeOriginC
          : exitEdgeOriginC;
        edgeDirection = useEntryFace
          ? entryEdgeDirectionC
          : exitEdgeDirectionC;
      }

      float rayAlong;
      float distanceToBarSquared = segmentSegmentDistanceSquared(
        ro,
        rayEnd,
        edgeOrigin,
        edgeDirection,
        rayAlong
      );
      if (distanceToBarSquared < nearestBarSquared) {
        nearestBarSquared = distanceToBarSquared;
        nearestAlong = rayAlong * wallT;
      }
    }

    float nearestBar = sqrt(nearestBarSquared);
    vec4 bounceLighting = uBounceLighting[bounce];
    vec3 barColor = bounceLighting.rgb;
    float depthLoss = bounceLighting.a;
    float airLoss = exp(-nearestAlong * 0.035);
    float opticalBloom = exp(-nearestBar * 42.0);
    radiance += throughput * depthLoss * airLoss *
      barColor * opticalBloom * 0.018;

    if (nearestBar < LIGHT_CORE_RADIUS) {
      float diffuser = 1.0 -
        smoothstep(0.008, LIGHT_CORE_RADIUS, nearestBar);
      float roundProfile = sqrt(max(
        0.0,
        1.0 -
          (nearestBar * nearestBar) /
          (LIGHT_CORE_RADIUS * LIGHT_CORE_RADIUS)
      ));
      vec3 tubeColor = mix(barColor, vec3(1.0), diffuser * 0.34);
      radiance += throughput * depthLoss * airLoss *
        tubeColor * (0.72 + roundProfile * 1.05);
      break;
    }

    if (nearestBar < LIGHT_RAIL_RADIUS) {
      float railBevel = smoothstep(0.018, 0.032, nearestBar);
      radiance += throughput * depthLoss *
        vec3(0.010, 0.012, 0.014) * railBevel;
      break;
    }

    vec3 hit = ro + rd * wallT;
    float edgeDistance = faceEdgeDistance(hit, faceIndex);
    if (edgeDistance < MIRROR_EDGE_INSET) {
      // This is a real break in the reflective panel, not a dark
      // overlay. Rays that miss the edge-mounted rail terminate in
      // the narrow structural channel between adjacent mirrors.
      float channelBevel = smoothstep(
        LIGHT_RAIL_RADIUS,
        MIRROR_EDGE_INSET,
        edgeDistance
      );
      radiance += throughput * depthLoss *
        mix(
          vec3(0.0025, 0.0030, 0.0035),
          vec3(0.009, 0.010, 0.011),
          channelBevel
        );
      break;
    }
    float seam = exp(-edgeDistance * 85.0);
    float faceVariation =
      0.88 + 0.12 * fract(float(faceIndex) * 0.618033);
    float grazing = pow(
      1.0 - abs(dot(faceNormal, -rd)),
      5.0
    );
    float reflectivity = mix(0.86, 0.935, grazing);

    vec3 coating = vec3(0.0045, 0.0052, 0.0062) * faceVariation;
    coating += vec3(0.006, 0.007, 0.008) * seam;
    radiance += throughput * coating * (1.0 - reflectivity) * 2.0;

    throughput *= reflectivity;
    throughput *= vec3(0.965, 0.978, 0.992);

    rd = reflect(rd, faceNormal);
    ro = hit - faceNormal * 0.0012;
    entryEdgeOriginA = exitEdgeOriginA;
    entryEdgeOriginB = exitEdgeOriginB;
    entryEdgeOriginC = exitEdgeOriginC;
    entryEdgeDirectionA = exitEdgeDirectionA;
    entryEdgeDirectionB = exitEdgeDirectionB;
    entryEdgeDirectionC = exitEdgeDirectionC;
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
  mat3 objectToWorld = uRotation;
  mat3 worldToObject = transpose(objectToWorld);
  vec3 ro = worldToObject * worldRo;
  vec3 rd = normalize(worldToObject * worldRd);

  vec3 color = background(worldRo, worldRd);
  float nearT = FAR;
  float farT = FAR;
  int nearFace = 0;
  int farFace = 0;
  bool glassHit = false;
  float frameT = FAR;
  vec3 frameNormal = vec3(0.0, 1.0, 0.0);
  bool frameHit = false;

  if (intersectsBoundingSphere(ro, rd)) {
    glassHit = intersectIcosahedron(
      ro,
      rd,
      nearT,
      farT,
      nearFace,
      farFace
    ) && nearT > 0.0;
    frameHit = intersectExteriorFrame(
      ro,
      rd,
      frameT,
      frameNormal
    );
  }

  if (
    frameHit &&
    (!glassHit || frameT < nearT + 0.035)
  ) {
    vec3 worldFrameNormal = normalize(
      objectToWorld * frameNormal
    );
    vec3 frameReflection = studioEnvironment(
      reflect(worldRd, worldFrameNormal)
    );
    float frameFacing = clamp(
      dot(-worldRd, worldFrameNormal),
      0.0,
      1.0
    );
    float frameFresnel =
      0.06 + 0.94 * pow(1.0 - frameFacing, 5.0);
    vec3 framePoint = ro + rd * frameT;
    float brushed = hash21(
      framePoint.xy * 740.0 +
      framePoint.z * 113.0
    );
    color =
      vec3(0.0035, 0.004, 0.0045) +
      frameReflection * (0.24 + frameFresnel * 0.44) +
      vec3(0.012, 0.013, 0.014) * brushed * 0.34;
  } else if (glassHit) {
    vec3 frontNormal = uPlanes[nearFace].xyz;
    vec3 frontHit = ro + rd * nearT;
    vec3 worldNormal = normalize(objectToWorld * frontNormal);
    vec3 reflectedWorld = reflect(worldRd, worldNormal);
    vec3 externalReflection = studioEnvironment(reflectedWorld);

    float facing = clamp(dot(-rd, frontNormal), 0.0, 1.0);
    float fresnel = 0.045 +
      (1.0 - 0.045) * pow(1.0 - facing, 5.0);

    vec3 insideOrigin = frontHit - frontNormal * 0.002;
    vec3 interior = traceMirroredInterior(
      insideOrigin,
      rd,
      nearFace
    );

    vec3 thinPanelTransmission = vec3(0.988, 0.993, 0.996);
    float coatingReflection = fresnel * 0.70;
    float transmission = (1.0 - fresnel) * 0.96;
    color =
      interior * thinPanelTransmission * transmission +
      externalReflection * coatingReflection;

    float edgeDistance = faceEdgeDistance(frontHit, nearFace);
    float hardEdge = 1.0 - smoothstep(0.004, 0.014, edgeDistance);
    float frameBevel =
      smoothstep(0.002, 0.006, edgeDistance) *
      (1.0 - smoothstep(0.010, 0.016, edgeDistance));
    float softEdge = exp(-edgeDistance * 38.0);
    vec3 edgeMetal =
      vec3(0.0045, 0.005, 0.0055) +
      externalReflection * (0.08 + frameBevel * 0.16) +
      vec3(0.018, 0.020, 0.022) * frameBevel;
    color = mix(color, edgeMetal, hardEdge * 0.95);
    color += externalReflection * softEdge * 0.012;

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

const POST_FRAGMENT_SHADER = `#version 300 es
precision highp float;

#define TEXTURE_SAMPLES_PER_PIXEL ${Math.max(
  1,
  Math.min(
    MAX_POST_PROCESS_TEXTURE_SAMPLES_PER_PIXEL,
    Math.round(POST_PROCESS_TEXTURE_SAMPLES_PER_PIXEL),
  ),
)}

out vec4 outColor;
in vec2 vUv;

uniform sampler2D uScene;
uniform vec2 uTexel;

vec3 brightSample(vec2 uv) {
  vec3 sampleColor = texture(uScene, uv).rgb;
  float brightness = max(
    sampleColor.r,
    max(sampleColor.g, sampleColor.b)
  );
  float threshold = smoothstep(0.52, 0.92, brightness);
  return sampleColor * threshold;
}

void main() {
  vec2 fromCenter = vUv - 0.5;
  vec2 chromaOffset = fromCenter * 0.00022;
  vec3 baseSample = texture(uScene, vUv).rgb;
  vec3 base = baseSample;
#if TEXTURE_SAMPLES_PER_PIXEL >= 2
  base.r = texture(uScene, vUv + chromaOffset).r;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 3
  base.b = texture(uScene, vUv - chromaOffset).b;
#endif

  vec3 bloom = vec3(0.0);
#if TEXTURE_SAMPLES_PER_PIXEL >= 4
  bloom += brightSample(vUv) * 0.08;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 5
  bloom += brightSample(vUv + vec2(uTexel.x * 2.0, 0.0)) * 0.08;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 6
  bloom += brightSample(vUv - vec2(uTexel.x * 2.0, 0.0)) * 0.08;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 7
  bloom += brightSample(vUv + vec2(0.0, uTexel.y * 2.0)) * 0.08;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 8
  bloom += brightSample(vUv - vec2(0.0, uTexel.y * 2.0)) * 0.08;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 9
  bloom += brightSample(vUv + uTexel * vec2(4.0, 4.0)) * 0.04;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 10
  bloom += brightSample(vUv + uTexel * vec2(-4.0, 4.0)) * 0.04;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 11
  bloom += brightSample(vUv + uTexel * vec2(4.0, -4.0)) * 0.04;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 12
  bloom += brightSample(vUv - uTexel * vec2(4.0, 4.0)) * 0.04;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 13
  bloom += brightSample(vUv + vec2(uTexel.x * 8.0, 0.0)) * 0.02;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 14
  bloom += brightSample(vUv - vec2(uTexel.x * 8.0, 0.0)) * 0.02;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 15
  bloom += brightSample(vUv + vec2(0.0, uTexel.y * 8.0)) * 0.02;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 16
  bloom += brightSample(vUv - vec2(0.0, uTexel.y * 8.0)) * 0.02;
#endif

  vec3 halation = vec3(
    bloom.r,
    bloom.r * 0.62,
    bloom.r * 0.34
  );
#if TEXTURE_SAMPLES_PER_PIXEL >= 17
  bloom += brightSample(vUv + vec2(uTexel.x * 16.0, 0.0)) * 0.012;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 18
  bloom += brightSample(vUv - vec2(uTexel.x * 16.0, 0.0)) * 0.012;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 19
  bloom += brightSample(vUv + vec2(0.0, uTexel.y * 16.0)) * 0.012;
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 20
  bloom += brightSample(vUv - vec2(0.0, uTexel.y * 16.0)) * 0.012;
#endif

  vec3 color = base + bloom * 0.72 + halation * 0.026;
  outColor = vec4(color, 1.0);
}`;

type Point = [number, number, number];

type GeometryData = {
  planes: Float32Array;
  faceEdgeOriginA: Float32Array;
  faceEdgeOriginB: Float32Array;
  faceEdgeOriginC: Float32Array;
  faceEdgeDirectionA: Float32Array;
  faceEdgeDirectionB: Float32Array;
  faceEdgeDirectionC: Float32Array;
  frameA: Float32Array;
  frameB: Float32Array;
  faceA: Float32Array;
  faceB: Float32Array;
  faceC: Float32Array;
};

function buildBounceLighting(): Float32Array {
  const lighting: number[] = [];
  const nearColor: Point = [1.0, 0.92, 0.82];
  const farColor: Point = [0.18, 0.58, 1.0];

  for (let bounce = 0; bounce < SHADER_MAX_REFLECTIONS; bounce++) {
    const depthPosition = Math.max(
      0,
      Math.min(1, (bounce - 1) / 14),
    );
    const depthMix =
      depthPosition *
      depthPosition *
      (3 - 2 * depthPosition) *
      0.82;
    lighting.push(
      nearColor[0] + (farColor[0] - nearColor[0]) * depthMix,
      nearColor[1] + (farColor[1] - nearColor[1]) * depthMix,
      nearColor[2] + (farColor[2] - nearColor[2]) * depthMix,
      Math.exp(-bounce * 0.064),
    );
  }

  return new Float32Array(lighting);
}

const BOUNCE_LIGHTING = buildBounceLighting();

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

  const edgeOrigins: number[] = [];
  const edgeDirections: number[] = [];
  const edgeKeys: string[] = [];
  const edgeIndexByKey = new Map<string, number>();
  const frameA: number[] = [];
  const frameB: number[] = [];
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      if (
        Math.abs(distance(vertices[i], vertices[j]) - edgeLength) <
        0.001
      ) {
        frameA.push(...vertices[i]);
        frameB.push(...vertices[j]);
        const a = vertices[i];
        const b = vertices[j];
        const trim = 0.035;
        const trimmedA: Point = [
          a[0] + (b[0] - a[0]) * trim,
          a[1] + (b[1] - a[1]) * trim,
          a[2] + (b[2] - a[2]) * trim,
        ];
        const trimmedB: Point = [
          b[0] + (a[0] - b[0]) * trim,
          b[1] + (a[1] - b[1]) * trim,
          b[2] + (a[2] - b[2]) * trim,
        ];
        const direction: Point = [
          trimmedB[0] - trimmedA[0],
          trimmedB[1] - trimmedA[1],
          trimmedB[2] - trimmedA[2],
        ];
        const lengthSquared =
          direction[0] * direction[0] +
          direction[1] * direction[1] +
          direction[2] * direction[2];
        const edgeKey = `${i}:${j}`;
        edgeOrigins.push(...trimmedA, 1 / lengthSquared);
        edgeDirections.push(...direction, lengthSquared);
        edgeIndexByKey.set(edgeKey, edgeKeys.length);
        edgeKeys.push(edgeKey);
      }
    }
  }

  const faceEdgeOriginA: number[] = [];
  const faceEdgeOriginB: number[] = [];
  const faceEdgeOriginC: number[] = [];
  const faceEdgeDirectionA: number[] = [];
  const faceEdgeDirectionB: number[] = [];
  const faceEdgeDirectionC: number[] = [];
  const appendFaceEdge = (
    first: number,
    second: number,
    origins: number[],
    directions: number[],
  ) => {
    const low = Math.min(first, second);
    const high = Math.max(first, second);
    const edgeIndex = edgeIndexByKey.get(`${low}:${high}`);
    if (edgeIndex === undefined) {
      throw new Error("Icosahedron face is missing an edge.");
    }
    origins.push(
      ...edgeOrigins.slice(edgeIndex * 4, edgeIndex * 4 + 4),
    );
    directions.push(
      ...edgeDirections.slice(edgeIndex * 4, edgeIndex * 4 + 4),
    );
  };
  for (const [a, b, c] of faces) {
    appendFaceEdge(
      a,
      b,
      faceEdgeOriginA,
      faceEdgeDirectionA,
    );
    appendFaceEdge(
      b,
      c,
      faceEdgeOriginB,
      faceEdgeDirectionB,
    );
    appendFaceEdge(
      c,
      a,
      faceEdgeOriginC,
      faceEdgeDirectionC,
    );
  }

  return {
    planes: new Float32Array(planes),
    faceEdgeOriginA: new Float32Array(faceEdgeOriginA),
    faceEdgeOriginB: new Float32Array(faceEdgeOriginB),
    faceEdgeOriginC: new Float32Array(faceEdgeOriginC),
    faceEdgeDirectionA: new Float32Array(faceEdgeDirectionA),
    faceEdgeDirectionB: new Float32Array(faceEdgeDirectionB),
    faceEdgeDirectionC: new Float32Array(faceEdgeDirectionC),
    frameA: new Float32Array(frameA),
    frameB: new Float32Array(frameB),
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

type Quaternion = readonly [
  number,
  number,
  number,
  number,
];

function multiplyQuaternions(
  a: Quaternion,
  b: Quaternion,
): Quaternion {
  return [
    a[3] * b[0] +
      a[0] * b[3] +
      a[1] * b[2] -
      a[2] * b[1],
    a[3] * b[1] -
      a[0] * b[2] +
      a[1] * b[3] +
      a[2] * b[0],
    a[3] * b[2] +
      a[0] * b[1] -
      a[1] * b[0] +
      a[2] * b[3],
    a[3] * b[3] -
      a[0] * b[0] -
      a[1] * b[1] -
      a[2] * b[2],
  ];
}

function normalizeQuaternion(
  quaternion: Quaternion,
): Quaternion {
  const inverseLength =
    1 /
    Math.hypot(
      quaternion[0],
      quaternion[1],
      quaternion[2],
      quaternion[3],
    );
  return [
    quaternion[0] * inverseLength,
    quaternion[1] * inverseLength,
    quaternion[2] * inverseLength,
    quaternion[3] * inverseLength,
  ];
}

function axisAngleQuaternion(
  x: number,
  y: number,
  z: number,
  angle: number,
): Quaternion {
  const halfAngle = angle * 0.5;
  const scale = Math.sin(halfAngle);
  return [
    x * scale,
    y * scale,
    z * scale,
    Math.cos(halfAngle),
  ];
}

function screenDragQuaternion(
  horizontal: number,
  vertical: number,
): Quaternion {
  const angle = Math.hypot(horizontal, vertical);
  if (angle < 1e-8) return [0, 0, 0, 1];
  const scale = Math.sin(angle * 0.5) / angle;

  // Pointer Y maps to the camera's horizontal axis; pointer X maps
  // to its vertical axis. These axes stay fixed on screen regardless
  // of the object's existing orientation.
  return [
    vertical * scale,
    horizontal * scale,
    0,
    Math.cos(angle * 0.5),
  ];
}

function slerpQuaternions(
  from: Quaternion,
  to: Quaternion,
  amount: number,
): Quaternion {
  let target = to;
  let cosine =
    from[0] * to[0] +
    from[1] * to[1] +
    from[2] * to[2] +
    from[3] * to[3];

  if (cosine < 0) {
    cosine = -cosine;
    target = [-to[0], -to[1], -to[2], -to[3]];
  }

  if (cosine > 0.9995) {
    return normalizeQuaternion([
      from[0] + (target[0] - from[0]) * amount,
      from[1] + (target[1] - from[1]) * amount,
      from[2] + (target[2] - from[2]) * amount,
      from[3] + (target[3] - from[3]) * amount,
    ]);
  }

  const angle = Math.acos(Math.min(1, cosine));
  const inverseSine = 1 / Math.sin(angle);
  const fromScale =
    Math.sin((1 - amount) * angle) * inverseSine;
  const toScale = Math.sin(amount * angle) * inverseSine;
  return [
    from[0] * fromScale + target[0] * toScale,
    from[1] * fromScale + target[1] * toScale,
    from[2] * fromScale + target[2] * toScale,
    from[3] * fromScale + target[3] * toScale,
  ];
}

function writeQuaternionMatrix(
  quaternion: Quaternion,
  matrix: Float32Array,
) {
  const [x, y, z, w] = quaternion;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const xw = x * w;
  const yw = y * w;
  const zw = z * w;

  matrix[0] = 1 - 2 * (yy + zz);
  matrix[1] = 2 * (xy + zw);
  matrix[2] = 2 * (xz - yw);
  matrix[3] = 2 * (xy - zw);
  matrix[4] = 1 - 2 * (xx + zz);
  matrix[5] = 2 * (yz + xw);
  matrix[6] = 2 * (xz + yw);
  matrix[7] = 2 * (yz - xw);
  matrix[8] = 1 - 2 * (xx + yy);
}

const INITIAL_ROTATION = multiplyQuaternions(
  axisAngleQuaternion(0, 1, 0, 0.54),
  axisAngleQuaternion(1, 0, 0, -0.16),
);

export default function MirrorChamber() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fpsCounterRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  const [rendererReady, setRendererReady] = useState(false);
  const controlsRef = useRef({
    dragging: false,
    pointerId: null as number | null,
    x: 0,
    y: 0,
    rotation: INITIAL_ROTATION,
    targetRotation: INITIAL_ROTATION,
    zoom: 5.55,
    targetZoom: 5.55,
    lastInteraction: 0,
  });

  useEffect(() => {
    // Strict Mode replays mount effects in development. Deferring readiness
    // lets that replay cancel the first setup before WebGL compiles anything.
    const initializationTimer = window.setTimeout(() => {
      setRendererReady(true);
    }, 0);

    return () => {
      window.clearTimeout(initializationTimer);
    };
  }, []);

  useEffect(() => {
    if (!rendererReady) return;

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

    let sceneProgram: WebGLProgram | null = null;
    let postProgram: WebGLProgram | null = null;
    let animationFrame = 0;
    let disposed = false;
    const startedAt = performance.now();
    let previousRenderAt = startedAt;
    let fpsSampleStartedAt = startedAt;
    let fpsFrameCount = 0;

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
      const postFragmentShader = compileShader(
        gl,
        gl.FRAGMENT_SHADER,
        POST_FRAGMENT_SHADER,
      );
      sceneProgram = gl.createProgram();
      postProgram = gl.createProgram();
      if (!sceneProgram || !postProgram) {
        throw new Error("Unable to create WebGL programs.");
      }
      const activeSceneProgram = sceneProgram;
      const activePostProgram = postProgram;
      gl.attachShader(activeSceneProgram, vertexShader);
      gl.attachShader(activeSceneProgram, fragmentShader);
      gl.linkProgram(activeSceneProgram);
      gl.attachShader(activePostProgram, vertexShader);
      gl.attachShader(activePostProgram, postFragmentShader);
      gl.linkProgram(activePostProgram);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      gl.deleteShader(postFragmentShader);

      if (!gl.getProgramParameter(activeSceneProgram, gl.LINK_STATUS)) {
        throw new Error(
          gl.getProgramInfoLog(activeSceneProgram) ??
            "Unable to link scene shaders.",
        );
      }
      if (!gl.getProgramParameter(activePostProgram, gl.LINK_STATUS)) {
        throw new Error(
          gl.getProgramInfoLog(activePostProgram) ??
            "Unable to link post-processing shaders.",
        );
      }

      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 3, -1, -1, 3]),
        gl.STATIC_DRAW,
      );
      const scenePosition = gl.getAttribLocation(
        activeSceneProgram,
        "aPosition",
      );
      const postPosition = gl.getAttribLocation(
        activePostProgram,
        "aPosition",
      );
      for (const position of [scenePosition, postPosition]) {
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(
          position,
          2,
          gl.FLOAT,
          false,
          0,
          0,
        );
      }
      gl.useProgram(activeSceneProgram);

      const uniform = (name: string) =>
        gl.getUniformLocation(activeSceneProgram, name);
      const uniforms = {
        resolution: uniform("uResolution"),
        time: uniform("uTime"),
        rotation: uniform("uRotation"),
        zoom: uniform("uZoom"),
        bounces: uniform("uBounces"),
        planes: uniform("uPlanes[0]"),
        faceEdgeOriginA: uniform("uFaceEdgeOriginA[0]"),
        faceEdgeOriginB: uniform("uFaceEdgeOriginB[0]"),
        faceEdgeOriginC: uniform("uFaceEdgeOriginC[0]"),
        faceEdgeDirectionA: uniform(
          "uFaceEdgeDirectionA[0]",
        ),
        faceEdgeDirectionB: uniform(
          "uFaceEdgeDirectionB[0]",
        ),
        faceEdgeDirectionC: uniform(
          "uFaceEdgeDirectionC[0]",
        ),
        frameA: uniform("uFrameA[0]"),
        frameB: uniform("uFrameB[0]"),
        faceA: uniform("uFaceA[0]"),
        faceB: uniform("uFaceB[0]"),
        faceC: uniform("uFaceC[0]"),
        bounceLighting: uniform("uBounceLighting[0]"),
      };
      const postUniforms = {
        scene: gl.getUniformLocation(activePostProgram, "uScene"),
        texel: gl.getUniformLocation(activePostProgram, "uTexel"),
      };
      const rotationMatrix = new Float32Array(9);

      const renderTexture = gl.createTexture();
      const framebuffer = gl.createFramebuffer();
      if (!renderTexture || !framebuffer) {
        throw new Error("Unable to create the photographic render target.");
      }
      gl.bindTexture(gl.TEXTURE_2D, renderTexture);
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_MIN_FILTER,
        gl.LINEAR,
      );
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_MAG_FILTER,
        gl.LINEAR,
      );
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_WRAP_S,
        gl.CLAMP_TO_EDGE,
      );
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_WRAP_T,
        gl.CLAMP_TO_EDGE,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        renderTexture,
        0,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const geometry = buildIcosahedron();
      gl.uniform4fv(uniforms.planes, geometry.planes);
      gl.uniform4fv(
        uniforms.faceEdgeOriginA,
        geometry.faceEdgeOriginA,
      );
      gl.uniform4fv(
        uniforms.faceEdgeOriginB,
        geometry.faceEdgeOriginB,
      );
      gl.uniform4fv(
        uniforms.faceEdgeOriginC,
        geometry.faceEdgeOriginC,
      );
      gl.uniform4fv(
        uniforms.faceEdgeDirectionA,
        geometry.faceEdgeDirectionA,
      );
      gl.uniform4fv(
        uniforms.faceEdgeDirectionB,
        geometry.faceEdgeDirectionB,
      );
      gl.uniform4fv(
        uniforms.faceEdgeDirectionC,
        geometry.faceEdgeDirectionC,
      );
      gl.uniform3fv(uniforms.frameA, geometry.frameA);
      gl.uniform3fv(uniforms.frameB, geometry.frameB);
      gl.uniform3fv(uniforms.faceA, geometry.faceA);
      gl.uniform3fv(uniforms.faceB, geometry.faceB);
      gl.uniform3fv(uniforms.faceC, geometry.faceC);
      gl.uniform4fv(uniforms.bounceLighting, BOUNCE_LIGHTING);

      const isCompact = window.matchMedia(
        "(max-width: 700px)",
      ).matches;
      if (isCompact) {
        controlsRef.current.zoom = 8.5;
        controlsRef.current.targetZoom = 8.5;
      }
      gl.uniform1i(
        uniforms.bounces,
        Math.max(
          1,
          Math.min(
            MAX_REFLECTIONS_PER_PIXEL,
            Math.round(REFLECTIONS_PER_PIXEL),
          ),
        ),
      );

      const resize = () => {
        const width = Math.max(
          1,
          Math.round(canvas.clientWidth * RENDER_PIXEL_RATIO),
        );
        const height = Math.max(
          1,
          Math.round(canvas.clientHeight * RENDER_PIXEL_RATIO),
        );
        if (
          canvas.width !== width ||
          canvas.height !== height
        ) {
          canvas.width = width;
          canvas.height = height;
          gl.bindTexture(gl.TEXTURE_2D, renderTexture);
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            width,
            height,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            null,
          );
          gl.bindTexture(gl.TEXTURE_2D, null);
          gl.viewport(0, 0, width, height);
        }
      };

      const render = (now: number) => {
        if (disposed || !sceneProgram || !postProgram) return;
        fpsFrameCount += 1;
        const fpsSampleDuration = now - fpsSampleStartedAt;
        if (fpsSampleDuration >= 500) {
          if (fpsCounterRef.current) {
            fpsCounterRef.current.textContent =
              `${Math.round(
                (fpsFrameCount * 1000) / fpsSampleDuration,
              )} FPS`;
          }
          fpsSampleStartedAt = now;
          fpsFrameCount = 0;
        }
        resize();
        const controls = controlsRef.current;
        const elapsedFrames =
          Math.max(0, now - previousRenderAt) /
          REFERENCE_FRAME_DURATION_MS;
        previousRenderAt = now;
        const rotationFollow =
          1 -
          Math.pow(
            1 - ROTATION_FOLLOW_PER_REFERENCE_FRAME,
            elapsedFrames,
          );
        controls.rotation = slerpQuaternions(
          controls.rotation,
          controls.targetRotation,
          rotationFollow,
        );
        controls.zoom +=
          (controls.targetZoom - controls.zoom) * 0.08;

        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(activeSceneProgram);
        gl.uniform2f(
          uniforms.resolution,
          canvas.width,
          canvas.height,
        );
        gl.uniform1f(
          uniforms.time,
          (now - startedAt) / 1000,
        );
        writeQuaternionMatrix(
          controls.rotation,
          rotationMatrix,
        );
        gl.uniformMatrix3fv(
          uniforms.rotation,
          false,
          rotationMatrix,
        );
        gl.uniform1f(uniforms.zoom, controls.zoom);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(activePostProgram);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, renderTexture);
        gl.uniform1i(postUniforms.scene, 0);
        gl.uniform2f(
          postUniforms.texel,
          1 / canvas.width,
          1 / canvas.height,
        );
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        animationFrame = window.requestAnimationFrame(render);
      };

      const pointerDown = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (
          controls.pointerId !== null ||
          !event.isPrimary ||
          (event.pointerType === "mouse" && event.button !== 0)
        ) {
          return;
        }

        event.preventDefault();
        controls.dragging = true;
        controls.pointerId = event.pointerId;
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add("is-dragging");
      };

      const pointerMove = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (
          !controls.dragging ||
          controls.pointerId !== event.pointerId
        ) {
          return;
        }

        const deltaX = event.clientX - controls.x;
        const deltaY = event.clientY - controls.y;
        const dragRotation = screenDragQuaternion(
          deltaX * 0.005,
          deltaY * 0.005,
        );

        // Pre-multiplication applies each drag around the camera's
        // screen axes, never around axes already rotated with the object.
        controls.targetRotation = normalizeQuaternion(
          multiplyQuaternions(
            dragRotation,
            controls.targetRotation,
          ),
        );
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastInteraction = performance.now();
      };

      const pointerUp = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (controls.pointerId !== event.pointerId) return;

        controls.dragging = false;
        controls.pointerId = null;
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
          1.72,
          Math.min(
            40,
            controls.targetZoom *
              Math.exp(event.deltaY * 0.001),
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
        gl.deleteFramebuffer(framebuffer);
        gl.deleteTexture(renderTexture);
        gl.deleteProgram(activeSceneProgram);
        gl.deleteProgram(activePostProgram);
      };
    } catch (caught) {
      const message =
        caught instanceof Error
          ? caught.message
          : "The renderer could not start.";
      console.error("[MirrorChamber renderer]", message);
      setError(message);
      return () => {
        disposed = true;
        window.cancelAnimationFrame(animationFrame);
        if (sceneProgram) gl.deleteProgram(sceneProgram);
        if (postProgram) gl.deleteProgram(postProgram);
      };
    }
  }, [rendererReady]);

  return (
    <main className="experience">
      <canvas
        ref={canvasRef}
        className="chamber"
        aria-label="A photorealistic interactive icosahedron with one-way mirrored faces and light bars along its interior edges"
      />
      <div
        ref={fpsCounterRef}
        className="fps-counter"
        aria-hidden="true"
      >
        -- FPS
      </div>
      {error ? (
        <div className="error-panel" role="alert">
          {error}
        </div>
      ) : null}
    </main>
  );
}
