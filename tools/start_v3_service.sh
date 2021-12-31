#!/usr/bin/env bash
# shellcheck disable=SC2006
check_conda=`which conda`
echo "check conda result are: $check_conda"
env_name=python375
python_version=3.7.5
conda_path="$HOME"/miniconda/bin/conda

if [[ -z $check_conda ]];
then
    wget "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p "$HOME"/miniconda
    # shellcheck disable=SC2046
    $conda_path init $(echo "$SHELL" | awk -F '/' '{print $NF}')
    echo 'Successfully installed miniconda...'
    echo -n 'Conda version: '
    $conda_path --version
    echo -e '\n'
else
    echo conda alerady install.
fi

# shellcheck disable=SC2006
conda_env_base=`$conda_path env list|grep base|awk '{print $3}'`
echo "conda env base : $conda_env_base"


python_dir=$conda_env_base"/envs"
files=$(ls "$python_dir")

python_interpreter=""
for filename in $files
do
   python_binary="$python_dir"/"$filename"/bin/python
   echo "$python_binary"
   # shellcheck disable=SC2006
   check_python_version=`$python_binary -c "import sys;print('.'.join([str(i) for i in sys.version_info[:3]]))"`
   echo "$python_binary version is $check_python_version"
   if [ "$check_python_version" = "$python_version" ];then
      python_interpreter=$python_binary
      break
   fi
done
echo check alerady install python exits python "$python_version" result:"$python_interpreter"

if [ "$python_interpreter" = "" ]; then
    echo begin install $env_name,excute conda create --name $env_name python=$python_version
    $conda_path create --name $env_name python=$python_version -y
    echo install python env_name over
    python_interpreter=$python_dir"/$env_name/bin/python"
fi

echo The path of the final python interpreter is "$python_interpreter"

if [ "$3" != "" ];then
  echo whl path is "$3"
  $python_interpreter -m pip install "$3"*.whl
fi


if [ "$1" = "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1
log=${cfg/yaml/log}
if [ "$2" = "data" ];then
  echo start data services
  $python_interpreter -u -m metis.data_svc.main "$cfg" >"$log" 2>&1
fi

if [ "$2" = "compute" ];then
  echo start data services
  $python_interpreter -u -m metis.data_svc.main "$cfg" >"$log" 2>&1
fi