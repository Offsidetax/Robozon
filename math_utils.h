#pragma once
#include <cmath>
#include <algorithm>

// Дополняем вашу структуру float3
struct float3 {
    float x, y, z;
    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};

struct Triangle {
    float3 v0, v1, v2;
    float3 centroid;
};

inline float3 operator-(const float3& a, const float3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(const float3& a, const float3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float3 normalize(const float3& v) {
    float invLen = 1.0f / std::sqrt(dot(v, v));
    return { v.x * invLen, v.y * invLen, v.z * invLen };
}

// Быстрое пересечение с AABB (Slab method)
inline bool intersectAABB(const float3& rayOrigin, const float3& rayDir,
    const float3& aabbMin, const float3& aabbMax, float closest_t) {
    float tx1 = (aabbMin.x - rayOrigin.x) / rayDir.x;
    float tx2 = (aabbMax.x - rayOrigin.x) / rayDir.x;
    float tmin = std::min(tx1, tx2);
    float tmax = std::max(tx1, tx2);

    float ty1 = (aabbMin.y - rayOrigin.y) / rayDir.y;
    float ty2 = (aabbMax.y - rayOrigin.y) / rayDir.y;
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));

    float tz1 = (aabbMin.z - rayOrigin.z) / rayDir.z;
    float tz2 = (aabbMax.z - rayOrigin.z) / rayDir.z;
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));

    return tmax >= tmin && tmin < closest_t && tmax > 0;
}

// Алгоритм Мёллера — Трумбора
inline bool intersectTriangle(const float3& rayOrigin, const float3& rayDir,
    const Triangle& tri, float& t) {
    const float EPSILON = 1e-8f;
    float3 edge1 = tri.v1 - tri.v0;
    float3 edge2 = tri.v2 - tri.v0;
    float3 h = cross(rayDir, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return false; // Луч параллелен треугольнику

    float f = 1.0f / a;
    float3 s = rayOrigin - tri.v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = f * dot(rayDir, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(edge2, q);
    return t > EPSILON;
}