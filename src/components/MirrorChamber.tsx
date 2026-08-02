"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// Rendering quality controls.
const MIRROR_BOUNCES = 24;
const POST_PROCESS_SAMPLES = 20;
const REFLECTION_FADE_RATE = 0.064;
const MOMENTUM_DECAY_MS = 200;
const FRAME_RADIUS = 0.043;
const ICOSAHEDRON_RADIUS = 1.56;
const SQUARE_VIEWPORT_DEFAULT_ZOOM = 5.55;
const DRAG_RADIANS_ACROSS_SHAPE = Math.PI;
const MIN_ZOOM = 1.72;
const MAX_ZOOM = 40;

const VERTEX_SHADER = `precision highp float;
in vec2 position;
out vec2 vUv;

void main() {
  vUv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `precision highp float;

out vec4 outColor;
in vec2 vUv;

uniform vec2 uResolution;
uniform float uTime;
uniform mat3 uRotation;
uniform float uZoom;
uniform vec4 uFaceEdgeOriginA[20];
uniform vec4 uFaceEdgeOriginB[20];
uniform vec4 uFaceEdgeOriginC[20];
uniform vec4 uFaceEdgeDirectionA[20];
uniform vec4 uFaceEdgeDirectionB[20];
uniform vec4 uFaceEdgeDirectionC[20];
uniform vec4 uBounceLighting[${MIRROR_BOUNCES}];

#define FACE_COUNT 20
#define MIRROR_BOUNCES ${MIRROR_BOUNCES}
#define FAR 100.0

const vec4 PLANES[FACE_COUNT] = vec4[FACE_COUNT](
  vec4(0.0, 0.934172359, 0.356822090, 1.239660977),
  vec4(0.0, 0.934172359, -0.356822090, 1.239660977),
  vec4(-0.577350269, 0.577350269, 0.577350269, 1.239660977),
  vec4(-0.577350269, 0.577350269, -0.577350269, 1.239660977),
  vec4(-0.934172359, 0.356822090, 0.0, 1.239660977),
  vec4(0.577350269, 0.577350269, 0.577350269, 1.239660977),
  vec4(0.577350269, 0.577350269, -0.577350269, 1.239660977),
  vec4(0.934172359, 0.356822090, 0.0, 1.239660977),
  vec4(0.0, -0.934172359, 0.356822090, 1.239660977),
  vec4(0.0, -0.934172359, -0.356822090, 1.239660977),
  vec4(-0.577350269, -0.577350269, 0.577350269, 1.239660977),
  vec4(-0.577350269, -0.577350269, -0.577350269, 1.239660977),
  vec4(-0.934172359, -0.356822090, 0.0, 1.239660977),
  vec4(0.577350269, -0.577350269, 0.577350269, 1.239660977),
  vec4(0.577350269, -0.577350269, -0.577350269, 1.239660977),
  vec4(0.934172359, -0.356822090, 0.0, 1.239660977),
  vec4(0.356822090, 0.0, 0.934172359, 1.239660977),
  vec4(-0.356822090, 0.0, 0.934172359, 1.239660977),
  vec4(0.356822090, 0.0, -0.934172359, 1.239660977),
  vec4(-0.356822090, 0.0, -0.934172359, 1.239660977)
);

const float LIGHT_CORE_RADIUS = 0.014;
const float MIRROR_EDGE_INSET = 0.043;
const float BOUNDING_RADIUS_SQUARED = 2.5921;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float edgeLineDistanceSquared(
  vec3 point,
  vec4 edgeOrigin,
  vec4 edgeDirection
) {
  vec3 offsetFromOrigin = point - edgeOrigin.xyz;
  float along = dot(
    offsetFromOrigin,
    edgeDirection.xyz
  ) * edgeOrigin.w;
  vec3 offset =
    offsetFromOrigin - edgeDirection.xyz * along;
  return dot(offset, offset);
}

vec2 raySegmentDistance(
  vec3 rayOrigin,
  vec3 rayDirection,
  float rayLength,
  vec4 edgeOriginData,
  vec4 edgeDirectionData
) {
  vec3 edgeOrigin = edgeOriginData.xyz;
  vec3 edgeDirection = edgeDirectionData.xyz;
  vec3 separationFromEdge = rayOrigin - edgeOrigin;
  float e = edgeDirectionData.w;
  float inverseE = edgeOriginData.w;
  float f = dot(edgeDirection, separationFromEdge);
  float c = dot(rayDirection, separationFromEdge);
  float b = dot(rayDirection, edgeDirection);
  float s;
  float t;
  float denominator = e - b * b;
  s = denominator != 0.0
    ? clamp(
        (b * f - c * e) / denominator,
        0.0,
        rayLength
      )
    : 0.0;
  t = (b * s + f) * inverseE;

  if (t < 0.0) {
    t = 0.0;
    s = clamp(-c, 0.0, rayLength);
  } else if (t > 1.0) {
    t = 1.0;
    s = clamp(b - c, 0.0, rayLength);
  }

  vec3 separation =
    (rayOrigin + rayDirection * s) -
    (edgeOrigin + edgeDirection * t);
  return vec2(dot(separation, separation), s);
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
  return sqrt(
    min(
      edgeLineDistanceSquared(
        point,
        uFaceEdgeOriginA[faceIndex],
        uFaceEdgeDirectionA[faceIndex]
      ),
      min(
        edgeLineDistanceSquared(
          point,
          uFaceEdgeOriginB[faceIndex],
          uFaceEdgeDirectionB[faceIndex]
        ),
        edgeLineDistanceSquared(
          point,
          uFaceEdgeOriginC[faceIndex],
          uFaceEdgeDirectionC[faceIndex]
        )
      )
    )
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
    vec3 normal = PLANES[i].xyz;
    float originSide = PLANES[i].w - dot(normal, ro);
    float directionSide = dot(normal, rd);

    if (abs(directionSide) < 0.00001) {
      if (originSide < 0.0) return false;
    } else if (directionSide < 0.0) {
      float numerator = -originSide;
      float denominator = -directionSide;
      if (numerator > nearT * denominator) {
        nearT = numerator / denominator;
        nearFace = i;
      }
    } else {
      if (originSide < farT * directionSide) {
        farT = originSide / directionSide;
        farFace = i;
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
    vec3 faceNormal = PLANES[i].xyz;
    float denominator = dot(faceNormal, rd);
    if (denominator > 0.00001) {
      float numerator =
        PLANES[i].w - dot(faceNormal, ro);
      if (
        numerator > 0.0002 * denominator &&
        numerator < nearest * denominator
      ) {
        nearest = numerator / denominator;
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
  float entryEdgeDistance = faceEdgeDistance(ro, entryFace);

  for (int bounce = 0; bounce < MIRROR_BOUNCES; bounce++) {

    vec3 faceNormal;
    int faceIndex;
    float wallT = intersectInterior(ro, rd, faceNormal, faceIndex);
    if (wallT >= FAR - 1.0) break;

    float nearestBarSquared = FAR * FAR;
    float nearestAlong = 0.0;
    vec4 exitEdgeOriginA = uFaceEdgeOriginA[faceIndex];
    vec4 exitEdgeOriginB = uFaceEdgeOriginB[faceIndex];
    vec4 exitEdgeOriginC = uFaceEdgeOriginC[faceIndex];
    vec4 exitEdgeDirectionA = uFaceEdgeDirectionA[faceIndex];
    vec4 exitEdgeDirectionB = uFaceEdgeDirectionB[faceIndex];
    vec4 exitEdgeDirectionC = uFaceEdgeDirectionC[faceIndex];
    vec2 candidate = raySegmentDistance(
      ro, rd, wallT,
      entryEdgeOriginA, entryEdgeDirectionA
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
    }
    candidate = raySegmentDistance(
      ro, rd, wallT,
      entryEdgeOriginB, entryEdgeDirectionB
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
    }
    candidate = raySegmentDistance(
      ro, rd, wallT,
      entryEdgeOriginC, entryEdgeDirectionC
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
    }
    candidate = raySegmentDistance(
      ro, rd, wallT,
      exitEdgeOriginA, exitEdgeDirectionA
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
    }
    candidate = raySegmentDistance(
      ro, rd, wallT,
      exitEdgeOriginB, exitEdgeDirectionB
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
    }
    candidate = raySegmentDistance(
      ro, rd, wallT,
      exitEdgeOriginC, exitEdgeDirectionC
    );
    if (candidate.x < nearestBarSquared) {
      nearestBarSquared = candidate.x;
      nearestAlong = candidate.y;
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

    if (
      bounce == 0 &&
      entryEdgeDistance < MIRROR_EDGE_INSET
    ) {
      break;
    }

    vec3 hit = ro + rd * wallT;
    float edgeDistance = faceEdgeDistance(hit, faceIndex);
    if (edgeDistance < MIRROR_EDGE_INSET) {
      // The inset is empty space between the light and mirror.
      // It receives no artificial rail or channel surface.
      break;
    }
    float seam = exp(-edgeDistance * 85.0);
    float faceVariation =
      0.88 + 0.12 * fract(float(faceIndex) * 0.618033);
    float grazingBase = 1.0 - abs(dot(faceNormal, -rd));
    float grazingSquared = grazingBase * grazingBase;
    float grazing =
      grazingSquared * grazingSquared * grazingBase;
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

  vec3 color;
  float nearT = FAR;
  float farT = FAR;
  int nearFace = 0;
  int farFace = 0;
  bool glassHit = false;
  float sceneDepth = 1.0;

  if (intersectsBoundingSphere(ro, rd)) {
    glassHit = intersectIcosahedron(
      ro,
      rd,
      nearT,
      farT,
      nearFace,
      farFace
    ) && nearT > 0.0;
  }

  if (glassHit) {
    vec3 frontNormal = PLANES[nearFace].xyz;
    vec3 frontHit = ro + rd * nearT;
    vec3 worldNormal = normalize(objectToWorld * frontNormal);
    vec3 reflectedWorld = reflect(worldRd, worldNormal);
    vec3 externalReflection = studioEnvironment(reflectedWorld);

    float facing = clamp(dot(-rd, frontNormal), 0.0, 1.0);
    float fresnelBase = 1.0 - facing;
    float fresnelSquared = fresnelBase * fresnelBase;
    float fresnelPower =
      fresnelSquared * fresnelSquared * fresnelBase;
    float fresnel =
      0.045 + (1.0 - 0.045) * fresnelPower;

    vec3 insideOrigin = frontHit - frontNormal * 0.002;
    vec3 interior = traceMirroredInterior(
      insideOrigin,
      rd,
      nearFace
    );

    vec3 thinPanelTransmission = vec3(0.988, 0.993, 0.996);
    float coatingReflection = fresnel * 0.70;
    float transmission = (1.0 - fresnel) * 0.96;
    vec3 mirroredPanel =
      interior * thinPanelTransmission * transmission +
      externalReflection * coatingReflection;
    float edgeDistance = faceEdgeDistance(frontHit, nearFace);
    float mirrorCoverage = smoothstep(
      MIRROR_EDGE_INSET - 0.004,
      MIRROR_EDGE_INSET,
      edgeDistance
    );
    color = mix(interior, mirroredPanel, mirrorCoverage);

    float silhouette = pow(1.0 - facing, 3.0);
    color += externalReflection * silhouette *
      (0.48 * mirrorCoverage);
    color += vec3(0.018, 0.020, 0.021) *
      ((1.0 - facing) * 0.34 * mirrorCoverage);

    const float depthNear = 0.1;
    const float depthFar = FAR;
    float cameraZ = worldRd.z * (nearT + 0.035);
    float depthA =
      (depthFar + depthNear) / (depthNear - depthFar);
    float depthB =
      (2.0 * depthFar * depthNear) /
      (depthNear - depthFar);
    sceneDepth =
      (depthA * cameraZ + depthB) / (-cameraZ) * 0.5 + 0.5;
  } else {
    color = background(worldRo, worldRd);
  }

  float vignette = dot(vUv - 0.5, vUv - 0.5);
  color *= 1.0 - vignette * 0.56;
  float grain =
    hash21(gl_FragCoord.xy + fract(uTime) * 719.31) - 0.5;
  color += grain * 0.0045;
  color = acesToneMap(color * 0.98);
  color = pow(color, vec3(0.4545));
  outColor = vec4(color, 1.0);
  gl_FragDepth = sceneDepth;
}`;

const FRAME_VERTEX_SHADER = `precision highp float;

in vec3 position;
in vec3 normal;

uniform vec2 uResolution;
uniform mat3 uRotation;
uniform float uZoom;

out vec3 vObjectPosition;
out vec3 vWorldPosition;
out vec3 vWorldNormal;

void main() {
  vec3 worldPosition = uRotation * position;
  vec3 cameraPosition =
    worldPosition - vec3(0.0, 0.10, uZoom);
  float aspect = uResolution.x / uResolution.y;
  float focalLength = 2.18 / 0.79;
  float depthNear = 0.1;
  float depthFar = 100.0;
  float depthA =
    (depthFar + depthNear) / (depthNear - depthFar);
  float depthB =
    (2.0 * depthFar * depthNear) /
    (depthNear - depthFar);
  float clipW = -cameraPosition.z;

  gl_Position = vec4(
    cameraPosition.x * focalLength / aspect,
    cameraPosition.y * focalLength,
    depthA * cameraPosition.z + depthB,
    clipW
  );
  vObjectPosition = position;
  vWorldPosition = worldPosition;
  vWorldNormal = uRotation * normal;
}`;

const FRAME_FRAGMENT_SHADER = `precision highp float;

in vec3 vObjectPosition;
in vec3 vWorldPosition;
in vec3 vWorldNormal;

uniform vec2 uResolution;
uniform float uTime;
uniform float uZoom;

out vec4 outColor;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
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
  vec3 normal = normalize(vWorldNormal);
  vec3 worldRay = normalize(
    vWorldPosition - vec3(0.0, 0.10, uZoom)
  );
  vec3 frameReflection = studioEnvironment(
    reflect(worldRay, normal)
  );
  float frameFacing = clamp(dot(-worldRay, normal), 0.0, 1.0);
  float frameFresnelBase = 1.0 - frameFacing;
  float frameFresnelSquared =
    frameFresnelBase * frameFresnelBase;
  float frameFresnel = 0.06 + 0.94 *
    frameFresnelSquared *
    frameFresnelSquared *
    frameFresnelBase;
  float brushed = hash21(
    vObjectPosition.xy * 740.0 +
    vObjectPosition.z * 113.0
  );
  vec3 color =
    vec3(0.0035, 0.004, 0.0045) +
    frameReflection * (0.24 + frameFresnel * 0.44) +
    vec3(0.012, 0.013, 0.014) * brushed * 0.34;

  vec2 uv = gl_FragCoord.xy / uResolution;
  float vignette = dot(uv - 0.5, uv - 0.5);
  color *= 1.0 - vignette * 0.56;
  float grain =
    hash21(gl_FragCoord.xy + fract(uTime) * 719.31) - 0.5;
  color += grain * 0.0045;
  color = acesToneMap(color * 0.98);
  color = pow(color, vec3(0.4545));
  outColor = vec4(color, 1.0);
}`;

const POST_FRAGMENT_SHADER = `precision highp float;

#define TEXTURE_SAMPLES_PER_PIXEL ${POST_PROCESS_SAMPLES}

out vec4 outColor;
in vec2 vUv;

uniform sampler2D uScene;
uniform vec2 uTexel;
uniform float uZoom;

vec3 brightSample(vec2 uv) {
  vec3 sampleColor = texture(uScene, uv).rgb;
  float brightness = max(
    sampleColor.r,
    max(sampleColor.g, sampleColor.b)
  );
  float threshold = smoothstep(0.52, 0.92, brightness);
  return sampleColor * threshold;
}

float luminance(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 antialiasedScene(vec2 uv) {
  vec3 center = texture(uScene, uv).rgb;
  vec3 north = texture(
    uScene,
    uv + vec2(0.0, uTexel.y)
  ).rgb;
  vec3 south = texture(
    uScene,
    uv - vec2(0.0, uTexel.y)
  ).rgb;
  vec3 east = texture(
    uScene,
    uv + vec2(uTexel.x, 0.0)
  ).rgb;
  vec3 west = texture(
    uScene,
    uv - vec2(uTexel.x, 0.0)
  ).rgb;

  float centerLuma = luminance(center);
  float northLuma = luminance(north);
  float southLuma = luminance(south);
  float eastLuma = luminance(east);
  float westLuma = luminance(west);
  float minimumLuma = min(
    centerLuma,
    min(min(northLuma, southLuma), min(eastLuma, westLuma))
  );
  float maximumLuma = max(
    centerLuma,
    max(max(northLuma, southLuma), max(eastLuma, westLuma))
  );
  float contrast = maximumLuma - minimumLuma;
  float threshold = max(0.035, maximumLuma * 0.12);
  float edgeBlend = smoothstep(
    threshold,
    threshold * 3.0,
    contrast
  ) * 0.90;

  float horizontalContrast = abs(eastLuma - westLuma);
  float verticalContrast = abs(northLuma - southLuma);
  vec3 acrossEdge = horizontalContrast > verticalContrast
    ? (east + west) * 0.5
    : (north + south) * 0.5;
  // Immediate neighbors create a one-pixel coverage transition,
  // rather than a wider image blur.
  return mix(
    center,
    (center + acrossEdge) * 0.5,
    edgeBlend
  );
}

bool canReceiveBloom() {
  vec2 screen = vUv * 2.0 - 1.0;
  screen.x *= uTexel.y / uTexel.x;
  vec3 rayOrigin = vec3(0.0, 0.10, uZoom);
  vec3 rayDirection = normalize(vec3(screen * 0.79, -2.18));
  float bloomReach =
    36.0 * uZoom * (0.79 / 2.18) * uTexel.y;
  float radius = 1.62 + bloomReach;
  float towardCenter = dot(rayOrigin, rayDirection);
  float originDistanceSquared =
    dot(rayOrigin, rayOrigin) - radius * radius;
  float discriminant =
    towardCenter * towardCenter - originDistanceSquared;
  return discriminant >= 0.0 &&
    (towardCenter < 0.0 || originDistanceSquared <= 0.0);
}

void main() {
  vec2 fromCenter = vUv - 0.5;
  vec2 chromaOffset = fromCenter * 0.00022;
  vec3 base = antialiasedScene(vUv);
#if TEXTURE_SAMPLES_PER_PIXEL >= 2
  base.r = mix(
    base.r,
    texture(uScene, vUv + chromaOffset).r,
    0.15
  );
#endif
#if TEXTURE_SAMPLES_PER_PIXEL >= 3
  base.b = mix(
    base.b,
    texture(uScene, vUv - chromaOffset).b,
    0.15
  );
#endif

  vec3 bloom = vec3(0.0);
  vec3 halation = vec3(0.0);
  if (canReceiveBloom()) {
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

    halation = vec3(
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
  }

  vec3 color = base + bloom * 0.72 + halation * 0.026;
  outColor = vec4(color, 1.0);
}`;

type Point = [number, number, number];

type GeometryData = {
  faceEdgeOriginA: Float32Array;
  faceEdgeOriginB: Float32Array;
  faceEdgeOriginC: Float32Array;
  faceEdgeDirectionA: Float32Array;
  faceEdgeDirectionB: Float32Array;
  faceEdgeDirectionC: Float32Array;
  frameVertices: Float32Array;
};

function buildBounceLighting(): Float32Array {
  const lighting: number[] = [];
  const nearColor: Point = [1.0, 0.92, 0.82];
  const farColor: Point = [0.18, 0.58, 1.0];

  for (let bounce = 0; bounce < MIRROR_BOUNCES; bounce++) {
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
      Math.exp(-bounce * REFLECTION_FADE_RATE),
    );
  }

  return new Float32Array(lighting);
}

const BOUNCE_LIGHTING = buildBounceLighting();

function normalizePoint(point: Point): Point {
  const inverseLength = 1 / Math.hypot(...point);
  return [
    point[0] * inverseLength,
    point[1] * inverseLength,
    point[2] * inverseLength,
  ];
}

function crossPoints(a: Point, b: Point): Point {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function appendFrameVertex(
  target: number[],
  position: Point,
  normal: Point,
) {
  target.push(...position, ...normal);
}

function appendFrameCylinder(
  target: number[],
  a: Point,
  b: Point,
) {
  const axis = normalizePoint([
    b[0] - a[0],
    b[1] - a[1],
    b[2] - a[2],
  ]);
  const reference: Point =
    Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const tangent = normalizePoint(crossPoints(axis, reference));
  const bitangent = crossPoints(axis, tangent);
  const radialSegments = 20;

  const radialAt = (angle: number): Point => [
    tangent[0] * Math.cos(angle) +
      bitangent[0] * Math.sin(angle),
    tangent[1] * Math.cos(angle) +
      bitangent[1] * Math.sin(angle),
    tangent[2] * Math.cos(angle) +
      bitangent[2] * Math.sin(angle),
  ];
  const offsetPoint = (point: Point, normal: Point): Point => [
    point[0] + normal[0] * FRAME_RADIUS,
    point[1] + normal[1] * FRAME_RADIUS,
    point[2] + normal[2] * FRAME_RADIUS,
  ];

  for (let segment = 0; segment < radialSegments; segment++) {
    const normalA = radialAt(
      (segment / radialSegments) * Math.PI * 2,
    );
    const normalB = radialAt(
      ((segment + 1) / radialSegments) * Math.PI * 2,
    );
    const a0 = offsetPoint(a, normalA);
    const a1 = offsetPoint(a, normalB);
    const b0 = offsetPoint(b, normalA);
    const b1 = offsetPoint(b, normalB);

    appendFrameVertex(target, a0, normalA);
    appendFrameVertex(target, b0, normalA);
    appendFrameVertex(target, b1, normalB);
    appendFrameVertex(target, a0, normalA);
    appendFrameVertex(target, b1, normalB);
    appendFrameVertex(target, a1, normalB);
  }
}

function appendFrameSphere(target: number[], center: Point) {
  const latitudeSegments = 12;
  const longitudeSegments = 24;
  const normalAt = (
    latitude: number,
    longitude: number,
  ): Point => {
    const latitudeAngle =
      -Math.PI * 0.5 +
      (latitude / latitudeSegments) * Math.PI;
    const longitudeAngle =
      (longitude / longitudeSegments) * Math.PI * 2;
    const latitudeRadius = Math.cos(latitudeAngle);
    return [
      latitudeRadius * Math.cos(longitudeAngle),
      Math.sin(latitudeAngle),
      latitudeRadius * Math.sin(longitudeAngle),
    ];
  };
  const positionAt = (normal: Point): Point => [
    center[0] + normal[0] * FRAME_RADIUS,
    center[1] + normal[1] * FRAME_RADIUS,
    center[2] + normal[2] * FRAME_RADIUS,
  ];

  for (
    let latitude = 0;
    latitude < latitudeSegments;
    latitude++
  ) {
    for (
      let longitude = 0;
      longitude < longitudeSegments;
      longitude++
    ) {
      const normal00 = normalAt(latitude, longitude);
      const normal01 = normalAt(latitude, longitude + 1);
      const normal10 = normalAt(latitude + 1, longitude);
      const normal11 = normalAt(latitude + 1, longitude + 1);
      const point00 = positionAt(normal00);
      const point01 = positionAt(normal01);
      const point10 = positionAt(normal10);
      const point11 = positionAt(normal11);

      appendFrameVertex(target, point00, normal00);
      appendFrameVertex(target, point10, normal10);
      appendFrameVertex(target, point11, normal11);
      appendFrameVertex(target, point00, normal00);
      appendFrameVertex(target, point11, normal11);
      appendFrameVertex(target, point01, normal01);
    }
  }
}

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

  const radius = ICOSAHEDRON_RADIUS;
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

  const edgeOrigins: number[] = [];
  const edgeDirections: number[] = [];
  const edgeKeys: string[] = [];
  const edgeIndexByKey = new Map<string, number>();
  const frameVertices: number[] = [];
  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      if (
        Math.abs(distance(vertices[i], vertices[j]) - edgeLength) <
        0.001
      ) {
        const a = vertices[i];
        const b = vertices[j];
        appendFrameCylinder(frameVertices, a, b);
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
  for (const vertex of vertices) {
    appendFrameSphere(frameVertices, vertex);
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
    faceEdgeOriginA: new Float32Array(faceEdgeOriginA),
    faceEdgeOriginB: new Float32Array(faceEdgeOriginB),
    faceEdgeOriginC: new Float32Array(faceEdgeOriginC),
    faceEdgeDirectionA: new Float32Array(faceEdgeDirectionA),
    faceEdgeDirectionB: new Float32Array(faceEdgeDirectionB),
    faceEdgeDirectionC: new Float32Array(faceEdgeDirectionC),
    frameVertices: new Float32Array(frameVertices),
  };
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
    angularVelocityX: 0,
    angularVelocityY: 0,
    lastPointerMoveAt: 0,
    zoom: SQUARE_VIEWPORT_DEFAULT_ZOOM,
    targetZoom: SQUARE_VIEWPORT_DEFAULT_ZOOM,
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

    let renderer: THREE.WebGLRenderer | null = null;
    let animationFrame = 0;
    let disposed = false;

    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        context: gl,
        alpha: false,
        antialias: false,
        depth: false,
        powerPreference: "high-performance",
      });
      const activeRenderer = renderer;
      activeRenderer.autoClear = false;
      activeRenderer.sortObjects = false;
      activeRenderer.outputColorSpace = THREE.LinearSRGBColorSpace;
      activeRenderer.debug.onShaderError = (
        shaderGl,
        program,
        vertexShader,
        fragmentShader,
      ) => {
        const messages = [
          shaderGl.getProgramInfoLog(program),
          shaderGl.getShaderInfoLog(vertexShader),
          shaderGl.getShaderInfoLog(fragmentShader),
        ].filter(Boolean);
        throw new Error(
          messages.join("\n") || "Unable to compile Three.js shaders.",
        );
      };

      const geometry = buildIcosahedron();
      const fullscreenGeometry = new THREE.BufferGeometry();
      fullscreenGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(
          new Float32Array([-1, -1, 3, -1, -1, 3]),
          2,
        ),
      );

      const frameGeometry = new THREE.BufferGeometry();
      const frameInterleaved = new THREE.InterleavedBuffer(
        geometry.frameVertices,
        6,
      );
      frameGeometry.setAttribute(
        "position",
        new THREE.InterleavedBufferAttribute(
          frameInterleaved,
          3,
          0,
          false,
        ),
      );
      frameGeometry.setAttribute(
        "normal",
        new THREE.InterleavedBufferAttribute(
          frameInterleaved,
          3,
          3,
          false,
        ),
      );
      frameGeometry.setDrawRange(
        0,
        geometry.frameVertices.length / 6,
      );

      const rotationMatrix = new Float32Array(9);
      const sceneRotation = new THREE.Matrix3();
      const frameRotation = new THREE.Matrix3();
      const sceneResolution = new THREE.Vector2();
      const frameResolution = new THREE.Vector2();
      const postTexel = new THREE.Vector2();

      const sceneMaterial = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        uniforms: {
          uResolution: { value: sceneResolution },
          uTime: { value: 0 },
          uRotation: { value: sceneRotation },
          uZoom: { value: SQUARE_VIEWPORT_DEFAULT_ZOOM },
          uFaceEdgeOriginA: {
            value: geometry.faceEdgeOriginA,
          },
          uFaceEdgeOriginB: {
            value: geometry.faceEdgeOriginB,
          },
          uFaceEdgeOriginC: {
            value: geometry.faceEdgeOriginC,
          },
          uFaceEdgeDirectionA: {
            value: geometry.faceEdgeDirectionA,
          },
          uFaceEdgeDirectionB: {
            value: geometry.faceEdgeDirectionB,
          },
          uFaceEdgeDirectionC: {
            value: geometry.faceEdgeDirectionC,
          },
          uBounceLighting: { value: BOUNCE_LIGHTING },
        },
        depthTest: true,
        depthWrite: true,
        depthFunc: THREE.AlwaysDepth,
        blending: THREE.NoBlending,
        toneMapped: false,
      });

      const frameMaterial = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: FRAME_VERTEX_SHADER,
        fragmentShader: FRAME_FRAGMENT_SHADER,
        uniforms: {
          uResolution: { value: frameResolution },
          uTime: { value: 0 },
          uRotation: { value: frameRotation },
          uZoom: { value: SQUARE_VIEWPORT_DEFAULT_ZOOM },
        },
        depthTest: true,
        depthWrite: true,
        depthFunc: THREE.LessDepth,
        side: THREE.DoubleSide,
        blending: THREE.NoBlending,
        toneMapped: false,
      });

      const postMaterial = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: VERTEX_SHADER,
        fragmentShader: POST_FRAGMENT_SHADER,
        uniforms: {
          uScene: { value: null },
          uTexel: { value: postTexel },
          uZoom: { value: SQUARE_VIEWPORT_DEFAULT_ZOOM },
        },
        depthTest: false,
        depthWrite: false,
        blending: THREE.NoBlending,
        toneMapped: false,
      });

      const sceneMesh = new THREE.Mesh(
        fullscreenGeometry,
        sceneMaterial,
      );
      const frameMesh = new THREE.Mesh(
        frameGeometry,
        frameMaterial,
      );
      const postMesh = new THREE.Mesh(
        fullscreenGeometry,
        postMaterial,
      );
      for (const mesh of [sceneMesh, frameMesh, postMesh]) {
        mesh.frustumCulled = false;
        mesh.matrixAutoUpdate = false;
      }

      const scenePass = new THREE.Scene();
      const framePass = new THREE.Scene();
      const postPass = new THREE.Scene();
      scenePass.matrixWorldAutoUpdate = false;
      framePass.matrixWorldAutoUpdate = false;
      postPass.matrixWorldAutoUpdate = false;
      scenePass.add(sceneMesh);
      framePass.add(frameMesh);
      postPass.add(postMesh);

      const camera = new THREE.Camera();
      camera.matrixAutoUpdate = false;
      camera.matrixWorldAutoUpdate = false;

      const renderTarget = new THREE.WebGLRenderTarget(1, 1, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false,
      });
      renderTarget.texture.generateMipmaps = false;
      renderTarget.texture.colorSpace = THREE.NoColorSpace;
      postMaterial.uniforms.uScene.value = renderTarget.texture;

      const getDefaultZoom = () =>
        SQUARE_VIEWPORT_DEFAULT_ZOOM *
        (canvas.clientHeight /
          Math.max(
            1,
            Math.min(canvas.clientWidth, canvas.clientHeight),
          ));
      controlsRef.current.zoom = getDefaultZoom();
      controlsRef.current.targetZoom = controlsRef.current.zoom;

      let renderWidth = 0;
      let renderHeight = 0;
      const resize = () => {
        const pixelRatio = Math.max(2, window.devicePixelRatio);
        const width = Math.max(
          1,
          Math.round(canvas.clientWidth * pixelRatio),
        );
        const height = Math.max(
          1,
          Math.round(canvas.clientHeight * pixelRatio),
        );
        if (width === renderWidth && height === renderHeight) return;

        renderWidth = width;
        renderHeight = height;
        activeRenderer.setSize(width, height, false);
        renderTarget.setSize(width, height);
        sceneResolution.set(width, height);
        frameResolution.set(width, height);
        postTexel.set(1 / width, 1 / height);
      };

      const startedAt = performance.now();
      let previousRenderAt = startedAt;
      let fpsSampleStartedAt = startedAt;
      let fpsFrameCount = 0;

      const render = (now: number) => {
        if (disposed) return;

        fpsFrameCount += 1;
        const fpsSampleDuration = now - fpsSampleStartedAt;
        if (fpsSampleDuration >= 500) {
          if (fpsCounterRef.current) {
            fpsCounterRef.current.textContent =
              Math.round(
                (fpsFrameCount * 1000) / fpsSampleDuration,
              ).toString() + " FPS";
          }
          fpsSampleStartedAt = now;
          fpsFrameCount = 0;
        }

        resize();
        const controls = controlsRef.current;
        const elapsedMilliseconds = Math.max(
          0,
          now - previousRenderAt,
        );
        previousRenderAt = now;

        if (
          !controls.dragging &&
          (Math.abs(controls.angularVelocityX) > 0.000001 ||
            Math.abs(controls.angularVelocityY) > 0.000001)
        ) {
          const momentumDecay = Math.exp(
            -elapsedMilliseconds / MOMENTUM_DECAY_MS,
          );
          const integratedTime =
            MOMENTUM_DECAY_MS * (1 - momentumDecay);
          const momentumRotation = screenDragQuaternion(
            controls.angularVelocityX * integratedTime,
            controls.angularVelocityY * integratedTime,
          );
          controls.rotation = normalizeQuaternion(
            multiplyQuaternions(
              momentumRotation,
              controls.rotation,
            ),
          );
          controls.angularVelocityX *= momentumDecay;
          controls.angularVelocityY *= momentumDecay;
        }
        controls.zoom +=
          (controls.targetZoom - controls.zoom) * 0.08;

        const elapsedSeconds = (now - startedAt) / 1000;
        writeQuaternionMatrix(
          controls.rotation,
          rotationMatrix,
        );
        sceneRotation.fromArray(rotationMatrix);
        frameRotation.fromArray(rotationMatrix);
        sceneMaterial.uniforms.uTime.value = elapsedSeconds;
        sceneMaterial.uniforms.uZoom.value = controls.zoom;
        frameMaterial.uniforms.uTime.value = elapsedSeconds;
        frameMaterial.uniforms.uZoom.value = controls.zoom;
        postMaterial.uniforms.uZoom.value = controls.zoom;

        activeRenderer.setRenderTarget(renderTarget);
        activeRenderer.clear(true, true, false);
        activeRenderer.render(scenePass, camera);
        activeRenderer.render(framePass, camera);

        activeRenderer.setRenderTarget(null);
        activeRenderer.render(postPass, camera);
        animationFrame = window.requestAnimationFrame(render);
      };

      const touchPointers = new Map<
        number,
        { x: number; y: number }
      >();
      let previousPinchDistance: number | null = null;

      const getPinchDistance = () => {
        const touches = Array.from(touchPointers.values());
        if (touches.length < 2) return null;
        return Math.hypot(
          touches[1].x - touches[0].x,
          touches[1].y - touches[0].y,
        );
      };

      const pointerDown = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (event.pointerType === "touch") {
          event.preventDefault();
          touchPointers.set(event.pointerId, {
            x: event.clientX,
            y: event.clientY,
          });
          canvas.setPointerCapture(event.pointerId);
          const now = performance.now();
          controls.lastInteraction = now;

          if (touchPointers.size === 1) {
            controls.dragging = true;
            controls.pointerId = event.pointerId;
            controls.x = event.clientX;
            controls.y = event.clientY;
            controls.angularVelocityX = 0;
            controls.angularVelocityY = 0;
            controls.lastPointerMoveAt = now;
            canvas.classList.add("is-dragging");
          } else {
            controls.dragging = false;
            controls.pointerId = null;
            controls.angularVelocityX = 0;
            controls.angularVelocityY = 0;
            previousPinchDistance = getPinchDistance();
            controls.targetZoom = controls.zoom;
            canvas.classList.remove("is-dragging");
          }
          return;
        }

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
        controls.angularVelocityX = 0;
        controls.angularVelocityY = 0;
        controls.lastPointerMoveAt = performance.now();
        controls.lastInteraction = controls.lastPointerMoveAt;
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add("is-dragging");
      };

      const pointerMove = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (event.pointerType === "touch") {
          if (!touchPointers.has(event.pointerId)) return;
          event.preventDefault();
          touchPointers.set(event.pointerId, {
            x: event.clientX,
            y: event.clientY,
          });

          if (touchPointers.size >= 2) {
            const pinchDistance = getPinchDistance();
            if (
              pinchDistance !== null &&
              previousPinchDistance !== null &&
              pinchDistance > 0
            ) {
              const zoom =
                controls.zoom *
                (previousPinchDistance / pinchDistance);
              controls.zoom = Math.max(
                MIN_ZOOM,
                Math.min(MAX_ZOOM, zoom),
              );
              controls.targetZoom = controls.zoom;
            }
            previousPinchDistance = pinchDistance;
            controls.lastInteraction = performance.now();
            return;
          }
        }

        if (
          !controls.dragging ||
          controls.pointerId !== event.pointerId
        ) {
          return;
        }

        const deltaX = event.clientX - controls.x;
        const deltaY = event.clientY - controls.y;
        const now = performance.now();
        const elapsedSinceMove = Math.max(
          1,
          now - controls.lastPointerMoveAt,
        );
        const cameraFocalLength = 2.18 / 0.79;
        const defaultShapeDiameter = Math.max(
          1,
          (canvas.clientHeight *
            ICOSAHEDRON_RADIUS *
            cameraFocalLength) /
            getDefaultZoom(),
        );
        const dragRadiansPerPixel =
          DRAG_RADIANS_ACROSS_SHAPE /
          defaultShapeDiameter;
        const horizontalRotation =
          deltaX * dragRadiansPerPixel;
        const verticalRotation =
          deltaY * dragRadiansPerPixel;
        const dragRotation = screenDragQuaternion(
          horizontalRotation,
          verticalRotation,
        );
        controls.rotation = normalizeQuaternion(
          multiplyQuaternions(
            dragRotation,
            controls.rotation,
          ),
        );
        const velocityBlend = Math.min(
          1,
          elapsedSinceMove / 20,
        );
        controls.angularVelocityX +=
          (horizontalRotation / elapsedSinceMove -
            controls.angularVelocityX) *
          velocityBlend;
        controls.angularVelocityY +=
          (verticalRotation / elapsedSinceMove -
            controls.angularVelocityY) *
          velocityBlend;
        controls.x = event.clientX;
        controls.y = event.clientY;
        controls.lastPointerMoveAt = now;
        controls.lastInteraction = now;
      };

      const pointerUp = (event: PointerEvent) => {
        const controls = controlsRef.current;
        if (event.pointerType === "touch") {
          if (!touchPointers.has(event.pointerId)) return;
          touchPointers.delete(event.pointerId);
          if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
          }
          const now = performance.now();
          controls.lastInteraction = now;

          if (touchPointers.size >= 2) {
            previousPinchDistance = getPinchDistance();
          } else if (touchPointers.size === 1) {
            const [remainingId, remainingTouch] =
              touchPointers.entries().next().value as [
                number,
                { x: number; y: number },
              ];
            previousPinchDistance = null;
            controls.dragging = true;
            controls.pointerId = remainingId;
            controls.x = remainingTouch.x;
            controls.y = remainingTouch.y;
            controls.angularVelocityX = 0;
            controls.angularVelocityY = 0;
            controls.lastPointerMoveAt = now;
            canvas.classList.add("is-dragging");
          } else {
            previousPinchDistance = null;
            controls.dragging = false;
            controls.pointerId = null;
            if (
              event.type === "pointercancel" ||
              now - controls.lastPointerMoveAt > 80
            ) {
              controls.angularVelocityX = 0;
              controls.angularVelocityY = 0;
            }
            canvas.classList.remove("is-dragging");
          }
          return;
        }

        if (controls.pointerId !== event.pointerId) return;

        controls.dragging = false;
        controls.pointerId = null;
        const now = performance.now();
        controls.lastInteraction = now;
        if (
          event.type === "pointercancel" ||
          now - controls.lastPointerMoveAt > 80
        ) {
          controls.angularVelocityX = 0;
          controls.angularVelocityY = 0;
        }
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        canvas.classList.remove("is-dragging");
      };

      const wheel = (event: WheelEvent) => {
        event.preventDefault();
        const controls = controlsRef.current;
        controls.targetZoom = Math.max(
          MIN_ZOOM,
          Math.min(
            MAX_ZOOM,
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
        canvas.removeEventListener("pointercancel", pointerUp);
        canvas.removeEventListener("wheel", wheel);
        sceneMaterial.dispose();
        frameMaterial.dispose();
        postMaterial.dispose();
        fullscreenGeometry.dispose();
        frameGeometry.dispose();
        renderTarget.dispose();
        activeRenderer.dispose();
      };
    } catch (caught) {
      disposed = true;
      window.cancelAnimationFrame(animationFrame);
      renderer?.dispose();
      const message =
        caught instanceof Error
          ? caught.message
          : "The Three.js renderer could not start.";
      console.error("[MirrorChamber renderer]", message);
      setError(message);
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
