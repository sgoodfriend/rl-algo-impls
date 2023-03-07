while getopts a:e:j:n:s: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        e) env=${OPTARG};;
        j) n_jobs=${OPTARG};;
        n) study_name=${OPTARG};;
        s) seeds=${OPTARG};;
    esac
done

TZ="America/Los_Angeles"
NOW=$(date +"%Y-%m-%dT%H:%M:%S")
study_name="${study_name:-$algo-$env-$NOW}"
STORAGE_PATH="sqlite:///runs/tuning.db"


mkdir -p runs
optuna create-study --study-name $study_name --storage $STORAGE_PATH --direction maximize --skip-if-exists

optimize () {
    for ((j=100;j<=n_jobs*100;j+=100)); do
        seed=()
        for ((s=0;s<seeds;s++)); do
            seed+="$((j+s*100/seeds)) "
        done
        echo python optimize.py --algo $algo --env $env --seed $seed --load-study --study-name $study_name --storage-path $STORAGE_PATH
    done
}

optimize | xargs -I CMD -P $n_jobs bash -c CMD