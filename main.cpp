#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "math_utils.h"

// --- Подключаем OpenCV ---
#include <opencv2/opencv.hpp>

struct alignas(32) BVHNode {
    float3 aabbMin;
    int leftFirst;
    float3 aabbMax;
    int triCount;
    bool isLeaf() const { return triCount > 0; }
};

class BVH {
public:
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    int nodesUsed = 0;

    void build() {
        for (auto& tri : triangles) {
            tri.centroid.x = (tri.v0.x + tri.v1.x + tri.v2.x) * 0.3333f;
            tri.centroid.y = (tri.v0.y + tri.v1.y + tri.v2.y) * 0.3333f;
            tri.centroid.z = (tri.v0.z + tri.v1.z + tri.v2.z) * 0.3333f;
        }

        bvhNodes.resize(triangles.size() * 2 - 1);
        BVHNode& root = bvhNodes[0];
        root.leftFirst = 0;
        root.triCount = static_cast<int>(triangles.size());
        nodesUsed = 1;

        updateNodeBounds(0);
        subdivide(0);
    }

private:
    void updateNodeBounds(int nodeIdx) {
        BVHNode& node = bvhNodes[nodeIdx];
        node.aabbMin = { 1e30f,  1e30f,  1e30f };
        node.aabbMax = { -1e30f, -1e30f, -1e30f };

        for (int i = 0; i < node.triCount; i++) {
            const Triangle& tri = triangles[node.leftFirst + i];
            node.aabbMin.x = std::min({ node.aabbMin.x, tri.v0.x, tri.v1.x, tri.v2.x });
            node.aabbMin.y = std::min({ node.aabbMin.y, tri.v0.y, tri.v1.y, tri.v2.y });
            node.aabbMin.z = std::min({ node.aabbMin.z, tri.v0.z, tri.v1.z, tri.v2.z });
            node.aabbMax.x = std::max({ node.aabbMax.x, tri.v0.x, tri.v1.x, tri.v2.x });
            node.aabbMax.y = std::max({ node.aabbMax.y, tri.v0.y, tri.v1.y, tri.v2.y });
            node.aabbMax.z = std::max({ node.aabbMax.z, tri.v0.z, tri.v1.z, tri.v2.z });
        }
    }

    void subdivide(int nodeIdx) {
        BVHNode& node = bvhNodes[nodeIdx];
        if (node.triCount <= 2) return;

        float3 extent = { node.aabbMax.x - node.aabbMin.x, node.aabbMax.y - node.aabbMin.y, node.aabbMax.z - node.aabbMin.z };
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        int first = node.leftFirst;
        int count = node.triCount;
        int mid = first + count / 2;

        std::nth_element(triangles.begin() + first, triangles.begin() + mid, triangles.begin() + first + count,
            [axis](const Triangle& a, const Triangle& b) { return a.centroid[axis] < b.centroid[axis]; });

        int leftChildIdx = nodesUsed++;
        int rightChildIdx = nodesUsed++;

        bvhNodes[leftChildIdx].leftFirst = first;
        bvhNodes[leftChildIdx].triCount = count / 2;
        updateNodeBounds(leftChildIdx);

        bvhNodes[rightChildIdx].leftFirst = mid;
        bvhNodes[rightChildIdx].triCount = count - (count / 2);
        updateNodeBounds(rightChildIdx);

        node.leftFirst = leftChildIdx;
        node.triCount = 0;

        subdivide(leftChildIdx);
        subdivide(rightChildIdx);
    }
};

void normalizeModel(std::vector<Triangle>& triangles) {
    if (triangles.empty()) return;

    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;

    for (const auto& tri : triangles) {
        float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
        for (int i = 0; i < 3; i++) {
            minX = std::min(minX, verts[i].x); maxX = std::max(maxX, verts[i].x);
            minY = std::min(minY, verts[i].y); maxY = std::max(maxY, verts[i].y);
            minZ = std::min(minZ, verts[i].z); maxZ = std::max(maxZ, verts[i].z);
        }
    }

    float offsetX = -(minX + maxX) * 0.5f;
    float offsetY = -(minY + maxY) * 0.5f;
    float offsetZ = -minZ; // Ставим на уровень конвейера (Z=0)

    for (auto& tri : triangles) {
        tri.v0.x += offsetX; tri.v0.y += offsetY; tri.v0.z += offsetZ;
        tri.v1.x += offsetX; tri.v1.y += offsetY; tri.v1.z += offsetZ;
        tri.v2.x += offsetX; tri.v2.y += offsetY; tri.v2.z += offsetZ;
    }
}

float traceRay(const float3& rayOrigin, const float3& rayDir, const BVH& bvh) {
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;
    float closest_t = 1e30f;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = bvh.bvhNodes[nodeIdx];

        if (!intersectAABB(rayOrigin, rayDir, node.aabbMin, node.aabbMax, closest_t)) continue;

        if (node.isLeaf()) {
            for (int i = 0; i < node.triCount; i++) {
                const Triangle& tri = bvh.triangles[node.leftFirst + i];
                float t;
                if (intersectTriangle(rayOrigin, rayDir, tri, t)) {
                    if (t < closest_t) closest_t = t;
                }
            }
        }
        else {
            stack[stackPtr++] = node.leftFirst + 1;
            stack[stackPtr++] = node.leftFirst;
        }
    }
    return closest_t;
}

// Структура для возврата габаритов
struct ScanResult {
    bool hit;
    float minX, maxX, minY, maxY, maxZ;
};

// Функция симуляции наклонной камеры
ScanResult renderDepthMap(int width, int height, const BVH& bvh) {
    std::vector<float> heightMap(width * height, 0.0f);
    const float PI = 3.14159265359f;
    float camHeight = 1100.0f;

    // ИСПРАВЛЕНИЕ: Оптическая ось отклонена на 60 градусов от вертикали
    float pitchRad = 60.0f * (PI / 180.0f);

    // ИСПРАВЛЕНИЕ: Полный вертикальный FOV: (80 - 60) * 2 = 40 градусов
    float fovDeg = 40.0f;

    // Вычисление cameraOrigin остается корректным, так как tan(60°) 
    // правильно отодвинет камеру по оси Y на ~1905 мм
    float3 cameraOrigin = { 0.0f, -camHeight * std::tan(pitchRad), camHeight };
    float3 forward = normalize(float3{ 0.0f, 0.0f, 0.0f } - cameraOrigin);
    float3 right = normalize(cross(forward, { 0.0f, 0.0f, 1.0f }));
    float3 up = cross(right, forward);

    float aspect = static_cast<float>(width) / height;
    float scale = std::tan((fovDeg * 0.5f) * (PI / 180.0f));

    ScanResult res = { false, 1e30f, -1e30f, 1e30f, -1e30f, -1e30f };

#pragma omp parallel
    {
        float l_minX = 1e30f, l_maxX = -1e30f, l_minY = 1e30f, l_maxY = -1e30f, l_maxZ = -1e30f;
        bool l_hit = false;

#pragma omp for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float ndcX = (2.0f * (x + 0.5f) / width) - 1.0f;
                float ndcY = 1.0f - (2.0f * (y + 0.5f) / height);
                float3 rayDir = normalize((ndcX * aspect * scale) * right + (ndcY * scale) * up + forward);

                float t = traceRay(cameraOrigin, rayDir, bvh);
                if (t < 1e29f) {
                    float3 hit = cameraOrigin + t * rayDir;
                    heightMap[y * width + x] = hit.z;
                    l_minX = std::min(l_minX, hit.x); l_maxX = std::max(l_maxX, hit.x);
                    l_minY = std::min(l_minY, hit.y); l_maxY = std::max(l_maxY, hit.y);
                    l_maxZ = std::max(l_maxZ, hit.z);
                    l_hit = true;
                }
            }
        }
#pragma omp critical
        {
            if (l_hit) {
                res.minX = std::min(res.minX, l_minX); res.maxX = std::max(res.maxX, l_maxX);
                res.minY = std::min(res.minY, l_minY); res.maxY = std::max(res.maxY, l_maxY);
                res.maxZ = std::max(res.maxZ, l_maxZ);
                res.hit = true;
            }
        }
    }

    std::ofstream file("height_output.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(heightMap.data()), heightMap.size() * sizeof(float));
    return res;
}

// === НОВАЯ ЛОГИКА OpenCV ===
// Ортографическое сканирование сверху вниз для получения чистого 2D сечения
bool checkCircularCrossSection(const BVH& bvh, const ScanResult& bounds) {
    // 1 пиксель = 1 мм. Добавляем рамку 10 мм для чистоты
    int minX = static_cast<int>(std::floor(bounds.minX)) - 10;
    int maxX = static_cast<int>(std::ceil(bounds.maxX)) + 10;
    int minY = static_cast<int>(std::floor(bounds.minY)) - 10;
    int maxY = static_cast<int>(std::ceil(bounds.maxY)) + 10;

    int width = maxX - minX;
    int height = maxY - minY;

    if (width <= 0 || height <= 0) return false;

    cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));

    // Сканируем строго сверху вниз (ортографическая проекция)
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float3 rayOrigin = { static_cast<float>(minX + x), static_cast<float>(minY + y), bounds.maxZ + 50.0f };
            float3 rayDir = { 0.0f, 0.0f, -1.0f }; // Луч бьет перпендикулярно ленте

            if (traceRay(rayOrigin, rayDir, bvh) < 1e29f) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // Морфологическое закрытие для устранения мелкого шума на границах
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Поиск внешних контуров
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    // Находим самый большой контур
    auto largestContour = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    // 1. Описанная окружность (R)
    cv::Point2f center;
    float R_out;
    cv::minEnclosingCircle(*largestContour, center, R_out);

    // 2. Вписанная окружность (r) через преобразование расстояний (Distance Transform)
    cv::Mat distTransform;
    cv::distanceTransform(mask, distTransform, cv::DIST_L2, 5);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(distTransform, &minVal, &maxVal, &minLoc, &maxLoc);
    float r_in = static_cast<float>(maxVal);

    if (R_out < 1.0f) return false;

    // 3. Вычисление коэффициента К
    float K = r_in / R_out;

    std::cout << "\n--- FORM ANALYSIS (OpenCV) ---\n";
    std::cout << "Radius of the circumcircle (R): " << R_out << " mm\n";
    std::cout << "Radius of the inscribed circle (r): " << r_in << " mm\n";
    std::cout << "Coefficient K (r/R): " << K << "\n";

    // Создаем отладочное изображение (очень поможет на защите проекта!)
    cv::Mat debugImg;
    cv::cvtColor(mask, debugImg, cv::COLOR_GRAY2BGR);
    cv::circle(debugImg, center, R_out, cv::Scalar(0, 0, 255), 2); // Красная - описанная
    cv::circle(debugImg, maxLoc, r_in, cv::Scalar(0, 255, 0), 2);  // Зеленая - вписанная
    cv::imwrite("cross_section_debug.png", debugImg);

    return K >= 0.8f;
}

// Загрузчик бинарного STL
bool loadBinarySTL(const std::string& filename, std::vector<Triangle>& outTriangles) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    char header[80];
    file.read(header, 80);
    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
    outTriangles.resize(numTriangles);
    for (uint32_t i = 0; i < numTriangles; i++) {
        file.seekg(12, std::ios::cur); // Пропускаем нормаль
        file.read(reinterpret_cast<char*>(&outTriangles[i].v0), 12);
        file.read(reinterpret_cast<char*>(&outTriangles[i].v1), 12);
        file.read(reinterpret_cast<char*>(&outTriangles[i].v2), 12);
        file.seekg(2, std::ios::cur);
    }
    return true;
}

int main() {
    BVH bvh;

    if (loadBinarySTL("test_item.stl", bvh.triangles)) {
        normalizeModel(bvh.triangles);
        std::cout << "Model loaded and normalized.\n";
    }
    else {
        std::cout << "STL not found.\n"; return -1;
    }

    std::cout << "Building BVH...\n";
    bvh.build();

    std::cout << "Rendering Depth Map...\n";
    ScanResult bounds = renderDepthMap(800, 600, bvh);

    if (bounds.hit) {
        float objLength = bounds.maxX - bounds.minX;
        float objWidth = bounds.maxY - bounds.minY;
        float objHeight = bounds.maxZ;

        float maxXY = std::max(objLength, objWidth);
        float minXY = std::min(objLength, objWidth);

        std::cout << "\n--- SCAN RESULTS ---\n";
        std::cout << "Facility dimensions (L x W x H): " << objLength << " x " << objWidth << " x " << objHeight << " мм\n";

        if (maxXY > 450.0f || minXY > 320.0f || objHeight > 320.0f) {
            std::cout << "STATUS: [DOES NOT FIT] Too large.\n";
        }
        else if (maxXY < 10.0f || minXY < 10.0f || objHeight < 10.0f) {
            std::cout << "STATUS: [DOES NOT FIT] Too small.\n";
        }
        else {
            // Если габариты в норме, запускаем проверку формы OpenCV
            bool isCircular = checkCircularCrossSection(bvh, bounds);
            if (isCircular) {
                std::cout << "STATUS: [REQUIRES REPACKAGING] Circular cross-section detected.\n";
            }
            else {
                std::cout << "STATUS: [SUITABLE FOR SORTING] Successful.\n";
            }
        }
    }
    return 0;
}