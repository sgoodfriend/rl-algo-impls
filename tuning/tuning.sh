ALGO=$1
ENV=$2
N_JOBS=$3

STUDY_NAME="$ALGO-$ENV"
STORAGE_PATH="sqlite://$STUDY_NAME.db"
optuna create-study --study-name $STUDY_NAME --storage $STORAGE_PATH --direction maximize

for s in {1..N_JOBS}
do
    python optimize.py --algo $ALGO --env $ENV --seed $s
done