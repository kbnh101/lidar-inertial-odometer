# CUDA residual tutorial

`data/source_point.txt`, `data/target_point.txt`를 읽고 CPU에서 대응점을 만든 뒤,
**각 pair의 residual만 CUDA로 계산**하는 작은 패키지입니다. Ceres, Jacobian,
ICP 반복, pose 최적화는 포함하지 않습니다. C++17, Eigen 3.4, CMake 3.18 이상,
CUDA Toolkit 및 NVIDIA GPU가 필요합니다. ROS 없이 빌드하거나 ROS 2의
`ament_cmake`/`colcon` 패키지로 사용할 수 있습니다.

## 계산 흐름

```text
CPU: common::load_point_cloud(source, target)
  -> common::KdTree3d(target), query = R * source + t
  -> pairs[i] = {원본 source 좌표, 최근접 target 좌표, target normal}
  -> cudaMalloc / cudaMemcpy(HostToDevice)
GPU: residual_kernel <<<ceil(N/256), 256>>>
  -> 스레드 i가 pair i의 residual 4개 계산
CPU: cudaMemcpy(DeviceToHost)
  -> 앞 5개 / RMSE 출력, 선택적으로 CSV 저장 및 CPU 기준값 비교
```

기존 hybrid ICP와 같은 두 종류의 **가중치 없는** residual을 구합니다.

```text
e       = R * source + t - target
r_point = [e.x, e.y, e.z]        (3차원, 부호 있는 벡터)
r_plane = target_normal dot e   (1차원, 부호 있는 거리)
```

`r_point`는 거리의 제곱이나 norm이 아닙니다. 콘솔 RMSE는 GPU에서 반환한
residual을 CPU에서 요약한 값입니다. Point RMSE는 `sqrt(mean(||r_point||²))`,
plane RMSE는 `sqrt(mean(r_plane²))`입니다.

기본 pose는 `R=I, t=0`입니다. `--pose tx ty tz roll pitch yaw`로 평가할 pose를
직접 지정할 수 있습니다. 회전은 `common::euler_zyx_to_rotation`의
`Rz(yaw) * Ry(pitch) * Rx(roll)`이며 각도 단위는 rad입니다.

텍스트 형식은 `x y z nx ny nz`이고 로더가 normal을 정규화합니다.
같은 행끼리 연결하지 않고 source마다 target의 1-NN을 찾습니다. 여러 source가
같은 target과 연결될 수 있으며, 거리 제한이나 normal 방향 일치 필터는 없습니다.
최근접 target의 normal이 0인 pair는 제외하고 개수를 출력합니다. Source normal은
계산에 사용하지 않습니다. 유효 pair가 하나도 없으면 오류로 종료합니다.

## 독립 빌드 및 실행

저장소 루트에서 실행합니다. `89`는 RTX 4060 기준이며 다른 GPU에서는 해당
architecture로 바꾸거나 옵션을 생략해 CUDA 컴파일러의 기본값을 사용하세요.

```bash
cmake -S cuda-residual-tutorial -B /tmp/cuda-residual-tutorial-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON
cmake --build /tmp/cuda-residual-tutorial-build -j2

# 기본 data를 읽어 모든 pair의 결과를 저장하고 CPU 계산값과 대조
/tmp/cuda-residual-tutorial-build/cuda_residual_demo \
  --verify --output /tmp/residuals.csv

# 특정 pose에서 residual 한 번 평가 (pose를 추정하는 기능은 없음)
/tmp/cuda-residual-tutorial-build/cuda_residual_demo \
  --source data/source_point.txt --target data/target_point.txt \
  --pose 0.1 -0.2 0.3 0.02 -0.03 0.04 --verify

ctest --test-dir /tmp/cuda-residual-tutorial-build --output-on-failure
```

CSV 열은 `source_index,target_index,rx,ry,rz,r_plane`입니다. Index는 주석과
빈 줄을 제외한 로딩된 cloud의 0-based index이며, CSV 행 순서는 source 순서입니다.
`--output`을 생략하면 파일을 만들지 않습니다. 지정한 출력 파일이 존재하면
덮어쓰되 입력 cloud 파일과 동일한 경로는 거부합니다.

## ROS 2 패키지로 실행

워크스페이스 루트(이 checkout을 `src/` 아래에 둔 디렉터리)에서:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select cuda_residual_tutorial \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON
source install/setup.bash
ros2 run cuda_residual_tutorial cuda_residual_demo --verify --output /tmp/residuals.csv
colcon test --packages-select cuda_residual_tutorial
colcon test-result --verbose
```

실행 파일은 ROS node가 아닌 콘솔 프로그램입니다. 설치된 실행 파일은
`share/cuda_residual_tutorial/data`에 함께 설치한 데이터를 우선 사용합니다.
독립 빌드 실행 파일은 checkout의 `data/`를 기본값으로 사용합니다. 두 경우 모두
현재 작업 디렉터리와 무관하게 실행할 수 있고 `--source`, `--target`으로 재정의합니다.

## 코드 읽는 순서

1. [`include/cuda_residual_tutorial/residual.hpp`](include/cuda_residual_tutorial/residual.hpp):
   CPU/GPU 사이에서 복사하는 `Pair`, row-major `Pose`, `Residual` 구조체.
2. [`app/main.cpp`](app/main.cpp): 기존 `common` 로더와 KD-tree로 pair 생성.
   검색에는 변환 좌표를 쓰지만 GPU에는 원본 source를 전달하는 부분에 주목하세요.
3. [`src/residual.cu`](src/residual.cu): GPU 메모리 할당 → 업로드 → kernel → 다운로드.
   `blockIdx.x * blockDim.x + threadIdx.x`로 pair index를 계산하고 마지막 블록에서
   `i >= count`인 스레드는 반환합니다. `double` 정밀도를 사용합니다.
4. [`test/test_residual.cpp`](test/test_residual.cpp): 수작업으로 계산 가능한 회전·이동,
   signed plane residual, 빈 입력, 256-thread 블록 경계와 출력 순서 검증.

`--verify`는 동일 pair에 대한 Eigen CPU 기준값과 CUDA 결과를 비교합니다.
각 성분의 허용 오차는 `1e-10 * (1 + abs(reference))`이며 불일치 시 비정상 종료합니다.
CTest는 이 검사를 실제 데이터의 identity pose와 회전·이동 pose에도 수행합니다.
GPU가 없거나 CUDA 호출이 실패하면 오류를 보고합니다.

학습을 위해 매 평가마다 메모리를 할당·해제하고 명시적으로 동기화합니다.
반복 실행 성능을 위한 버퍼 재사용, 비동기 전송, GPU reduction은 생략했습니다.
작은 데이터에서 CPU보다 빠르다는 성능 보장은 없습니다.
