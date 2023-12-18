# franka_simulation

* install
  * `conda env create f environment.yaml`

* C++ 파일 빌드 방법
  1. conda 가상환경 활성화
  2. `franka_simulation` 디렉토리로 이동
  3. `chmod +x build_cpp.sh`
  4. `./build_cpp.sh`
  * 해당 쉘 스크립트를 실행하면 py_src 내의 파일을 실행할 준비 과정이 완료됩니다.

* Use
  * `franka_simulation/py_src` 디렉토리로 이동
  *  train_test_fullaction.py : full action 학습 or test
  *  train_test_rpy.py : Orientation 학습 or test

* 인자 설명 - train.py, train_test_fullaction.py
  1. --exec : 실행 모드, train or eval 중 선택하여 입력
  2. ex. python train.py --exec train

* 인자 설명 - train_test_rpy.py
  1. --exec : 실행 모드, train or eval 중 선택하여 입력
  2. --env  : 실행 환경, [6d_train, 6d_test, 3d_test] 중 선택하여 입력
  3. ex. python train_test_rpy.py --exec eval --env 6d_test