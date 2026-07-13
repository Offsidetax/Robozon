#include <iostream>
#include <fstream>
#include <vector>
#include "math_utils.h"
#include <algorithm>
#include <cmath>

// Выравнивание до 32 байт (одна кэш-линия обычно 64 байта, так что ровно 2 узла на линию)
struct alignas(32) BVHNode {
    float3 aabbMin;
    int leftFirst; // Для листа: индекс первого треугольника. Для узла: индекс левого потомка.
    float3 aabbMax;
    int triCount;  // Если > 0, это лист. Если 0, это внутренний узел.

    bool isLeaf() const { return triCount > 0; }
};

class BVH {
public:
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    int nodesUsed = 0;

    void build() {
        // Предвычисляем центроиды для всех треугольников (один раз!)
        for (auto& tri : triangles) {
            tri.centroid.x = (tri.v0.x + tri.v1.x + tri.v2.x) * 0.3333f;
            tri.centroid.y = (tri.v0.y + tri.v1.y + tri.v2.y) * 0.3333f;
            tri.centroid.z = (tri.v0.z + tri.v1.z + tri.v2.z) * 0.3333f;
        }

        // Выделяем память с запасом: в бинарном дереве максимум (2 * N - 1) узлов
        bvhNodes.resize(triangles.size() * 2 - 1);

        // Инициализируем корневой узел
        BVHNode& root = bvhNodes[0];
        root.leftFirst = 0;
        root.triCount = static_cast<int>(triangles.size());
        nodesUsed = 1;

        updateNodeBounds(0);
        subdivide(0);
    }

private:
    // Функция расчета Bounding Box (AABB) для конкретного узла
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

    // Рекурсивное разделение по медиане
    void subdivide(int nodeIdx) {
        BVHNode& node = bvhNodes[nodeIdx];

        // Критерий остановки: если треугольников 2 или меньше, оставляем лист
        if (node.triCount <= 2) return;

        // 1. Ищем самую длинную ось (X=0, Y=1, Z=2)
        float3 extent = {
            node.aabbMax.x - node.aabbMin.x,
            node.aabbMax.y - node.aabbMin.y,
            node.aabbMax.z - node.aabbMin.z
        };

        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        // 2. Определяем границы для std::nth_element
        int first = node.leftFirst;
        int count = node.triCount;
        int mid = first + count / 2;

        // 3. Разделяем массив треугольников за O(N)
        std::nth_element(
            triangles.begin() + first,
            triangles.begin() + mid,
            triangles.begin() + first + count,
            [axis](const Triangle& a, const Triangle& b) {
                return a.centroid[axis] < b.centroid[axis];
            }
        );

        // 4. Резервируем два новых узла в линейном массиве
        int leftChildIdx = nodesUsed++;
        int rightChildIdx = nodesUsed++;

        // 5. Инициализируем левого потомка
        bvhNodes[leftChildIdx].leftFirst = first;
        bvhNodes[leftChildIdx].triCount = count / 2;
        updateNodeBounds(leftChildIdx);

        // 6. Инициализируем правого потомка
        bvhNodes[rightChildIdx].leftFirst = mid;
        bvhNodes[rightChildIdx].triCount = count - (count / 2);
        updateNodeBounds(rightChildIdx);

        // 7. Превращаем текущий узел во внутренний
        node.leftFirst = leftChildIdx; // Правый потомок всегда лежит сразу за левым (leftChildIdx + 1)
        node.triCount = 0;             // triCount == 0 - маркер внутреннего узла

        // 8. Рекурсия
        subdivide(leftChildIdx);
        subdivide(rightChildIdx);
    }
};

// Обновленная сигнатура traceRay (принимает BVH по ссылке)
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

// Константа для перевода градусов в радианы
const float PI = 3.14159265359f;

// Обновленная функция рендеринга с OpenMP и расчетом габаритов
void renderDepthMap(int width, int height, const BVH& bvh) {
    std::vector<float> heightMap(width * height, 0.0f);

    float camHeight = 1100.0f;
    float pitchAngleDeg = 30.0f;
    float fovDeg = 60.0f;
    float pitchRad = pitchAngleDeg * (PI / 180.0f);

    float3 cameraOrigin = { 0.0f, -camHeight * std::tan(pitchRad), camHeight };
    float3 cameraTarget = { 0.0f, 0.0f, 0.0f };
    float3 globalUp = { 0.0f, 0.0f, 1.0f };

    float3 forward = normalize(cameraTarget - cameraOrigin);
    float3 right = normalize(cross(forward, globalUp));
    float3 up = cross(right, forward);

    float aspect = static_cast<float>(width) / height;
    float scale = std::tan((fovDeg * 0.5f) * (PI / 180.0f));

    // Глобальные переменные для физических габаритов (в миллиметрах)
    float globalMinX = 1e30f, globalMinY = 1e30f;
    float globalMaxX = -1e30f, globalMaxY = -1e30f, globalMaxZ = -1e30f;
    bool hitAnything = false;

    // Включаем многопоточность OpenMP
#pragma omp parallel
    {
        // Локальные переменные для каждого потока (чтобы избежать конфликтов доступа к памяти)
        float localMinX = 1e30f, localMinY = 1e30f;
        float localMaxX = -1e30f, localMaxY = -1e30f, localMaxZ = -1e30f;
        bool localHit = false;

#pragma omp for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float ndcX = (2.0f * (x + 0.5f) / width) - 1.0f;
                float ndcY = 1.0f - (2.0f * (y + 0.5f) / height);
                float3 rayDir = normalize((ndcX * aspect * scale) * right + (ndcY * scale) * up + forward);

                float t = traceRay(cameraOrigin, rayDir, bvh);

                if (t < 1e29f) {
                    float3 hitPoint = cameraOrigin + t * rayDir;
                    heightMap[y * width + x] = hitPoint.z;

                    // Обновляем границы товара
                    localMinX = std::min(localMinX, hitPoint.x);
                    localMaxX = std::max(localMaxX, hitPoint.x);
                    localMinY = std::min(localMinY, hitPoint.y);
                    localMaxY = std::max(localMaxY, hitPoint.y);
                    localMaxZ = std::max(localMaxZ, hitPoint.z);
                    localHit = true;
                }
            }
        }

        // Безопасно объединяем локальные данные потоков в глобальные переменные
#pragma omp critical
        {
            if (localHit) {
                globalMinX = std::min(globalMinX, localMinX);
                globalMaxX = std::max(globalMaxX, localMaxX);
                globalMinY = std::min(globalMinY, localMinY);
                globalMaxY = std::max(globalMaxY, localMaxY);
                globalMaxZ = std::max(globalMaxZ, localMaxZ);
                hitAnything = true;
            }
        }
    }

    // Сохраняем результат
    std::ofstream file("height_output.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(heightMap.data()), heightMap.size() * sizeof(float));

    // Анализ габаритов и вывод результата
    if (hitAnything) {
        float objLength = globalMaxX - globalMinX;
        float objWidth = globalMaxY - globalMinY;
        float objHeight = globalMaxZ; // Высота считается от ленты (0.0)

        std::cout << "\n--- RESULTS OF SCAN ---\n";
        std::cout << "Dimensions of object (L x W x H): "
            << objLength << " x " << objWidth << " x " << objHeight << " mm\n";

        // Ориентация на ленте может быть любой, поэтому сортируем длину и ширину для проверки
        float maxXY = std::max(objLength, objWidth);
        float minXY = std::min(objLength, objWidth);

        // Проверка по правилам хакатона
        if (maxXY > 450.0f || minXY > 320.0f || objHeight > 320.0f) {
            std::cout << "STATUS: [REJECTED] Exceeds maximum dimensions (450x320x320)\n";
        }
        else if (maxXY < 10.0f || minXY < 10.0f || objHeight < 10.0f) {
            std::cout << "STATUS: [REJECTED] Item is smaller than minimum dimensions (10x10x10)\n";
        }
        else {
            std::cout << "STATUS: [PRE-APPROVED] Dimensions within limits. Form check required.\n";
        }
        std::cout << "-------------------------------\n";
    }
    else {
        std::cout << "STATUS: The feed is empty; no products were found.\n";
    }
}

// Быстрый загрузчик бинарного STL
bool loadBinarySTL(const std::string& filename, std::vector<Triangle>& outTriangles) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    char header[80];
    file.read(header, 80); // Пропускаем заголовок

    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));

    outTriangles.resize(numTriangles);
    for (uint32_t i = 0; i < numTriangles; i++) {
        float normal[3];
        file.read(reinterpret_cast<char*>(&normal), 12);
        file.read(reinterpret_cast<char*>(&outTriangles[i].v0), 12);
        file.read(reinterpret_cast<char*>(&outTriangles[i].v1), 12);
        file.read(reinterpret_cast<char*>(&outTriangles[i].v2), 12);

        uint16_t attributeByteCount;
        file.read(reinterpret_cast<char*>(&attributeByteCount), 2); // Пропускаем атрибуты
    }
    return true;
}

// Функция для автоматического выравнивания модели на ленте (Z=0) и центрирования (X=0, Y=0)
void normalizeModel(std::vector<Triangle>& triangles) {
    if (triangles.empty()) return;

    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;

    // 1. Находим габариты исходной CAD-модели
    for (const auto& tri : triangles) {
        float3 verts[3] = { tri.v0, tri.v1, tri.v2 };
        for (int i = 0; i < 3; i++) {
            minX = std::min(minX, verts[i].x); maxX = std::max(maxX, verts[i].x);
            minY = std::min(minY, verts[i].y); maxY = std::max(maxY, verts[i].y);
            minZ = std::min(minZ, verts[i].z); maxZ = std::max(maxZ, verts[i].z);
        }
    }

    // 2. Вычисляем смещение: центр X и Y в 0, а самый низ (minZ) поднимаем на 0 (уровень ленты)
    float offsetX = -(minX + maxX) * 0.5f;
    float offsetY = -(minY + maxY) * 0.5f;
    float offsetZ = -minZ;

    // 3. Сдвигаем все вершины модели в пространстве
    for (auto& tri : triangles) {
        tri.v0.x += offsetX; tri.v0.y += offsetY; tri.v0.z += offsetZ;
        tri.v1.x += offsetX; tri.v1.y += offsetY; tri.v1.z += offsetZ;
        tri.v2.x += offsetX; tri.v2.y += offsetY; tri.v2.z += offsetZ;
    }
}

int main() {
    BVH bvh;

    if (loadBinarySTL("test_item.stl", bvh.triangles)) {
        std::cout << "Loaded " << bvh.triangles.size() << " triangles.\n";

        // ВЫЗЫВАЕМ НОРМАЛИЗАЦИЮ ПЕРЕД ПОСТРОЕНИЕМ BVH
        normalizeModel(bvh.triangles);
        std::cout << "Model normalized (centered and placed on Z=0).\n";

    }
    else {
        std::cout << "STL not found. Using fallback triangle.\n";
        bvh.triangles.push_back({ {-50, -50, 0}, {50, -50, 0}, {0, 50, 0} });
    }

    std::cout << "Building BVH...\n";
    bvh.build();
    std::cout << "BVH Nodes used: " << bvh.nodesUsed << "\n";

    std::cout << "Rendering depth map...\n";
    renderDepthMap(800, 600, bvh);
    std::cout << "Done! Output saved to height_output.bin\n";

    return 0;
}