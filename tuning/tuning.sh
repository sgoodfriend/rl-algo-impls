while getopts a:e:n:s: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        e) env=${OPTARG};;
        n) n_jobs=${OPTARG};;
        s) study_name=${OPTARG};;
    esac
done

TZ="America/Los_Angeles"
NOW=$(date +"%Y-%m-%dT%H:%M:%S")
study_name="${study_name:-$algo-$env-$NOW}"
STORAGE_PATH="sqlite:///runs/tuning.db"

mkdir -p runs
optuna create-study --study-name $study_name --storage $STORAGE_PATH --direction maximize --skip-if-exists

optimize () {
    for ((s=100;s<=n_jobs*100;s+=100)); do
        echo python optimize.py --algo $algo --env $env --seed $s --load-study --study-name $study_name --storage-path $STORAGE_PATH
    done
}

optimize | xargs -I CMD -P $n_jobs bash -c CMD