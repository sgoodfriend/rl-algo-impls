ALGO=$1
ENV=$2
N_JOBS=$3

TZ="America/Los_Angeles"
NOW=$(date +"%Y-%m-%dT%H:%M:%S")
STUDY_NAME="$ALGO-$ENV-$NOW"
STORAGE_PATH="sqlite:///runs/tuning.db"

mkdir -p runs
optuna create-study --study-name $STUDY_NAME --storage $STORAGE_PATH --direction maximize

optimize () {
    for ((s=1;s<=N_JOBS;s++)); do
        echo python optimize.py --algo $ALGO --env $ENV --seed $s --load-study --study-name $STUDY_NAME --storage-path $STORAGE_PATH
    done
}

optimize | xargs -I CMD -P $N_JOBS bash -c CMD