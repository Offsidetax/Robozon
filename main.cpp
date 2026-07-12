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

// Обновленная функция рендеринга с моделью наклонной камеры
void renderDepthMap(int width, int height, const BVH& bvh) {
    std::vector<float> heightMap(width * height, 0.0f); // Заполняем нулями (уровень ленты)

    // 1. Физические параметры установки (в миллиметрах и градусах)
    float camHeight = 1100.0f;     // Высота по вашей схеме
    float pitchAngleDeg = 30.0f;   // Угол наклона от вертикали (30 градусов)
    float fovDeg = 60.0f;          // Угол обзора камеры (Field of View)

    // Перевод угла в радианы
    float pitchRad = pitchAngleDeg * (PI / 180.0f);

    // 2. Вычисление позиции камеры
    // Сдвигаем камеру назад по оси Y (вдоль конвейера), чтобы при наклоне она смотрела в центр (0,0,0)
    float3 cameraOrigin = { 0.0f, -camHeight * std::tan(pitchRad), camHeight };
    float3 cameraTarget = { 0.0f, 0.0f, 0.0f }; // Точка на ленте конвейера
    float3 globalUp = { 0.0f, 0.0f, 1.0f }; // Ось Z смотрит вверх

    // 3. Построение базиса камеры (Look-At)
    float3 forward = normalize(cameraTarget - cameraOrigin);
    float3 right = normalize(cross(forward, globalUp));
    float3 up = cross(right, forward);

    // 4. Масштаб проекции экрана
    float aspect = static_cast<float>(width) / height;
    float scale = std::tan((fovDeg * 0.5f) * (PI / 180.0f));

    // 5. Трассировка
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // NDC координаты [-1, 1]
            float ndcX = (2.0f * (x + 0.5f) / width) - 1.0f;
            float ndcY = 1.0f - (2.0f * (y + 0.5f) / height);

            // Направление луча в мировом пространстве
            float3 rayDir = normalize((ndcX * aspect * scale) * right + (ndcY * scale) * up + forward);

            // Трассируем луч
            float t = traceRay(cameraOrigin, rayDir, bvh);

            // 6. Магия перевода: вычисляем физическую координату Z (высоту над лентой)
            if (t < 1e29f) { // Если попали в объект
                float3 hitPoint = cameraOrigin + t * rayDir; // Точка пересечения в 3D
                heightMap[y * width + x] = hitPoint.z;       // Сохраняем абсолютную ВЫСОТУ в мм
            }
        }
    }

    // Сохраняем карту ВЫСОТ
    std::ofstream file("height_output.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(heightMap.data()), heightMap.size() * sizeof(float));
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

int main() {
    BVH bvh;

    // Загрузите тестовую модель объекта сортировки (замените на свой файл)
    // Если файла нет, можно захардкодить один тестовый треугольник для проверки
    if (loadBinarySTL("test_item.stl", bvh.triangles)) {
        std::cout << "Loaded " << bvh.triangles.size() << " triangles.\n";
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
    std::cout << "Done! Output saved to depth_output.bin\n";

    return 0;
}