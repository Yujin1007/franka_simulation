# franka_simulation

* install
  * conda install --file packagelist.txt
  * conda env create --file environment.yaml

* C++ 파일 빌드 방법
  * `chmod +x build_cpp.sh`
  * `./scripts/build_cpp.sh`
  * 해당 쉘 스크립트를 실행하면 py_src 내의 파일을 실행할 준비 과정이 완료됩니다.
  * controller / task planning 수정 시 simulate 폴더 내부의 필요한 코드를 수정 한후 빌드 해 주세요.

* Use
  *  train.py : evaluation
  *  train.py --exec train : train
