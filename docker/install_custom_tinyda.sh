env_path=$1
tda_path=$2

cdir="$(pwd)"

cd $cdir
source ${env_path}bin/activate
mkdir ${tda_path}
cd ${tda_path}
git clone https://github.com/bagr-sus/tinyDA
cd tinyDA
git checkout shared-archive-proposals
#python -m pip install -e . #editable
python -m pip install . #non-editable
cd $cdir