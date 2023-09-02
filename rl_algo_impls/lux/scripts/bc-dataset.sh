mkdir -p data/lux
cd data/lux

create_dataset() {
    if [ -d $1 ]; then
        echo "Dataset $1 already exists"
        return
    fi
    mkdir $1
    cd $1
    local kaggle_name="${2:-$1}"
    kaggle datasets download -d sgoodfriend/$kaggle_name
    unzip $kaggle_name.zip
    cd ..
}

create_dataset lux-replays-deimos-npz lux-replays-submission-deimos-npz
create_dataset lux-replays-flg-npz
create_dataset lux-replays-ry_andy_-npz lux-replays-ry-andy-npz
create_dataset lux-replays-tigga-npz
create_dataset lux-replays-siestaguru-npz
