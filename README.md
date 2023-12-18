# franka_simulation

* install
  * `conda env create -f environment.yaml`

* C++ 파일 빌드 방법
  1. conda 가상환경 활성화
  2. `franka_simulation` 디렉토리로 이동
  3. `chmod +x build_cpp.sh`
  4. `./build_cpp.sh`
  * 해당 쉘 스크립트를 실행하면 py_src 내의 파일을 실행할 준비 과정이 완료됩니다.
  * controller / task planning 수정 시 simulate 폴더 내부의 필요한 코드를 수정 한 후 빌드 해 주세요.
  * Eigen 관련 오류 발생시 [fatal error: Eigen/Core: No such file or directory]
     * `sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen`
     * (eigen 경로 확인요망 출처: https://velog.io/@ys__us/fatal-error-EigenCore-No-such-file-or-directory) 

* Use
  * `franka_simulation/py_src` 디렉토리로 이동
  *  python franka_valve.py

* 인자 설명 - franka_valve.py
  1. --exec : 실행 모드, train or eval 중 선택하여 입력
  2. ex. python train.py --exec train
