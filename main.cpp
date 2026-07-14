#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "math_utils.h"
#include <windows.h>
#include <omp.h>

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

bool checkMaskForCircle(const cv::Mat& mask, float& K_out) {
    cv::Mat morphMask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, morphMask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morphMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    auto largestContour = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    // Описанная окружность (R)
    cv::Point2f center;
    float R_out;
    cv::minEnclosingCircle(*largestContour, center, R_out);

    // Защита от шумовых контуров
    if (R_out < 5.0f) return false;

    // Вписанная окружность (r)
    cv::Mat distTransform;
    cv::distanceTransform(morphMask, distTransform, cv::DIST_L2, 5);
    double minVal, maxVal;
    cv::minMaxLoc(distTransform, &minVal, &maxVal);
    float r_in = static_cast<float>(maxVal);

    K_out = r_in / R_out;
    return K_out >= 0.8f;
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

// Функция симуляции 3D-камеры: возвращает облако точек
std::vector<float3> scanPointCloud(int width, int height, const BVH& bvh, float3 cameraOrigin, float pitchRad) {
    std::vector<float3> cloud;
    const float PI = 3.14159265359f;
    float fovDeg = 40.0f; // Вертикальный угол обзора

    float3 forward = normalize(float3{ 0.0f, 0.0f, 0.0f } - cameraOrigin);
    float3 right = normalize(cross(forward, { 0.0f, 0.0f, 1.0f }));
    float3 up = cross(right, forward);

    float aspect = static_cast<float>(width) / height;
    float scale = std::tan((fovDeg * 0.5f) * (PI / 180.0f));

#pragma omp parallel
    {
        // Локальный вектор для каждого потока, чтобы избежать блокировок
        std::vector<float3> local_cloud;
        local_cloud.reserve((width * height) / omp_get_num_threads());

#pragma omp for nowait
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float ndcX = (2.0f * (x + 0.5f) / width) - 1.0f;
                float ndcY = 1.0f - (2.0f * (y + 0.5f) / height);
                float3 rayDir = normalize((ndcX * aspect * scale) * right + (ndcY * scale) * up + forward);

                float t = traceRay(cameraOrigin, rayDir, bvh);
                if (t < 1e29f) {
                    local_cloud.push_back(cameraOrigin + t * rayDir);
                }
            }
        }

        // Сливаем локальные векторы в общий массив
#pragma omp critical
        {
            cloud.insert(cloud.end(), local_cloud.begin(), local_cloud.end());
        }
    }
    return cloud;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // --- ДОБАВЛЕННЫЙ БЛОК ---
    BVH bvh;
    std::cout << "Loading STL...\n";
    // Замени "model.stl" на реальный путь к твоему тестовому файлу
    if (!loadBinarySTL("test_item.stl", bvh.triangles)) {
        std::cout << "ERROR: Failed to load STL file.\n";
        return -1;
    }
    // ------------------------

    std::cout << "Building BVH...\n";
    bvh.build();

    std::cout << "Scanning Object via Dual Cameras...\n";

    // Параметры установки камер
    const float PI = 3.14159265359f;
    float camHeight = 1100.0f;
    float pitchRad = 60.0f * (PI / 180.0f);
    float yOffset = camHeight * std::tan(pitchRad);

    // Камера 1 (спереди) и Камера 2 (сзади)
    float3 cam1Origin = { 0.0f, -yOffset, camHeight };
    float3 cam2Origin = { 0.0f,  yOffset, camHeight };

    std::vector<float3> cloud1 = scanPointCloud(800, 600, bvh, cam1Origin, pitchRad);
    std::vector<float3> cloud2 = scanPointCloud(800, 600, bvh, cam2Origin, pitchRad);

    // --- ДОБАВЛЕННЫЙ БЛОК: СОХРАНЕНИЕ ДЛЯ PYTHON ---
    std::cout << "Saving point clouds for Python visualization...\n";
    std::ofstream out1("cloud1.bin", std::ios::binary);
    if (out1) out1.write(reinterpret_cast<const char*>(cloud1.data()), cloud1.size() * sizeof(float3));
    out1.close();

    std::ofstream out2("cloud2.bin", std::ios::binary);
    if (out2) out2.write(reinterpret_cast<const char*>(cloud2.data()), cloud2.size() * sizeof(float3));
    out2.close();
    // -----------------------------------------------

    // Объединяем облака
    std::vector<float3> full_cloud;
    full_cloud.reserve(cloud1.size() + cloud2.size());
    full_cloud.insert(full_cloud.end(), cloud1.begin(), cloud1.end());
    full_cloud.insert(full_cloud.end(), cloud2.begin(), cloud2.end());

    if (full_cloud.empty()) {
        std::cout << "ERROR: Object not detected on the belt.\n";
        return -1;
    }
    std::cout << "Generated Point Cloud: " << full_cloud.size() << " points.\n";

    // 1. Подготовка данных для PCA ИЗ ОБЛАКА ТОЧЕК (а не из STL)
    int numPoints = full_cloud.size();
    cv::Mat data_pts(numPoints, 3, CV_32FC1);
    for (int i = 0; i < numPoints; ++i) {
        data_pts.at<float>(i, 0) = full_cloud[i].x;
        data_pts.at<float>(i, 1) = full_cloud[i].y;
        data_pts.at<float>(i, 2) = full_cloud[i].z;
    }

    // 2. Расчет PCA
    cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    float3 center = { pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1), pca.mean.at<float>(0, 2) };

    std::vector<float3> axes(3);
    for (int i = 0; i < 3; ++i) {
        axes[i] = { pca.eigenvectors.at<float>(i, 0), pca.eigenvectors.at<float>(i, 1), pca.eigenvectors.at<float>(i, 2) };
    }

    // 3. Вычисление габаритов Oriented Bounding Box (OBB)
    float minExt[3] = { 1e30f, 1e30f, 1e30f };
    float maxExt[3] = { -1e30f, -1e30f, -1e30f };

    for (int i = 0; i < numPoints; ++i) {
        float3 pt = { data_pts.at<float>(i, 0), data_pts.at<float>(i, 1), data_pts.at<float>(i, 2) };
        float3 d = pt - center;
        for (int j = 0; j < 3; ++j) {
            float proj = dot(d, axes[j]);
            minExt[j] = std::min(minExt[j], proj);
            maxExt[j] = std::max(maxExt[j], proj);
        }
    }

    std::vector<float> dims = { maxExt[0] - minExt[0], maxExt[1] - minExt[1], maxExt[2] - minExt[2] };
    std::sort(dims.rbegin(), dims.rend()); // Сортируем: Length, Width, Height

    std::cout << "\n--- DIMENSIONS (OBB) ---\n";
    std::cout << "Extracted dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << " mm\n";

    // ПРИОРИТЕТ 1: Габаритный контроль
    if (dims[0] > 450.0f || dims[1] > 320.0f || dims[2] > 320.0f) {
        std::cout << "STATUS: [NOT SUITABLE FOR DIMENSIONAL SORTING] The object is too large.\n";
        return 0;
    }
    if (dims[2] < 10.0f) {
        std::cout << "STATUS: [NOT SUITABLE FOR SIZE-BASED SORTING] The object is too small.\n";
        return 0;
    }

    std::cout << "\n--- FORM ANALYSIS (Cross-Section Slicing) ---\n";
    bool requiresRepackaging = false;

    // Настройки слайсера
    const int NUM_SLICES = 10; // Количество слоев (сечений) вдоль каждой оси
    const int MIN_POINTS_IN_SLICE = 50; // Отсекаем слои, где слишком мало точек (шум)

    for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
        if (requiresRepackaging) break; // Прерываем внешний цикл, если круг уже найден

        int uIdx = (axisIdx + 1) % 3;
        int vIdx = (axisIdx + 2) % 3;

        // Вычисляем толщину одного слоя
        float sliceThickness = (maxExt[axisIdx] - minExt[axisIdx]) / NUM_SLICES;

        // Защита от деления на ноль для абсолютно плоских объектов
        if (sliceThickness < 1.0f) continue;

        int width = static_cast<int>(std::ceil(maxExt[uIdx] - minExt[uIdx])) + 20;
        int height = static_cast<int>(std::ceil(maxExt[vIdx] - minExt[vIdx])) + 20;

        // Создаем массив масок (стопку сечений) для текущей оси
        std::vector<cv::Mat> masks(NUM_SLICES);
        for (int i = 0; i < NUM_SLICES; ++i) {
            masks[i] = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }

        // 1-Й ПРОХОД: Распределяем все точки облака по соответствующим слоям (Сложность O(N))
        for (const auto& pt : full_cloud) {
            float3 d = pt - center;

            // localW - это глубина точки вдоль оси нарезки
            float localW = dot(d, axes[axisIdx]);
            float localU = dot(d, axes[uIdx]) - minExt[uIdx] + 10.0f;
            float localV = dot(d, axes[vIdx]) - minExt[vIdx] + 10.0f;

            // Вычисляем индекс слоя, в который попадает точка
            int sliceIdx = static_cast<int>((localW - minExt[axisIdx]) / sliceThickness);

            // Защита (clamping) индекса массива
            sliceIdx = std::max(0, std::min(sliceIdx, NUM_SLICES - 1));

            int x = std::max(0, std::min(static_cast<int>(localU), width - 1));
            int y = std::max(0, std::min(static_cast<int>(localV), height - 1));

            // Закрашиваем пиксель в конкретном слое
            masks[sliceIdx].at<uchar>(y, x) = 255;
        }

        // 2-Й ПРОХОД: Анализ каждого слоя с помощью морфологии
        for (int i = 0; i < NUM_SLICES; ++i) {
            // Оптимизация: пропускаем пустые или почти пустые слои
            if (cv::countNonZero(masks[i]) < MIN_POINTS_IN_SLICE) {
                continue;
            }

            float K = 0.0f;
            if (checkMaskForCircle(masks[i], K)) {
                std::cout << "Axis [" << axisIdx << "] Slice [" << i + 1 << "/" << NUM_SLICES << "] K = " << K << " -> CIRCLE DETECTED\n";
                requiresRepackaging = true;
                break; // Круг найден - дальше текущую ось не проверяем
            }
            else {
                // Опциональный вывод успешных слоев (полезно для отладки)
                std::cout << "Axis [" << axisIdx << "] Slice [" << i + 1 << "/" << NUM_SLICES << "] K = " << K << " -> OK\n";
            }
        }
    }

    // Итоговый вывод статуса
    if (requiresRepackaging) {
        std::cout << "\nSTATUS: [NOT SUITABLE FOR SORTING WITHOUT ADDITIONAL PACKAGING]\n";
        std::cout << "REASON: Circular cross-section detected in object profile.\n";
    }
    else {
        std::cout << "\nSTATUS: [SUITABLE FOR SORTING]\n";
    }

    return 0;
}