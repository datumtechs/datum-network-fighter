image=local/rosetta:v0.5.0

# build
docker build -t $image .

# run
scripts_path=$(cd $(dirname $0); pwd)
docker run \
    --name build_rosetta \
    -v $scripts_path/dist:/opt/dist \
    --rm \
    $image cp -r /opt/Rosetta/dist /opt
