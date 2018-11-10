#!/bin/bash
#SBATCH -A ccnsb
#SBATCH -p long
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

function kill_process
{
# Getting the PID of the process
PID=$1

# Number of seconds to wait before using "kill -9"
WAIT_SECONDS=10

# Counter to keep count of how many seconds have passed
count=0

while kill -2 $PID > /dev/null
do
    # Wait for one second
    sleep 1
    # Increment the second counter
    ((count++))

    # Has the process been killed? If so, exit the loop.
    if ! pgrep "python" | grep $PID > /dev/null ; then
        break
    fi

    # Have we exceeded $WAIT_SECONDS? If so, kill the process with "kill -9"
    # and exit the loop
    if [ $count -gt $WAIT_SECONDS ]; then
        kill -9 $PID
        break
    fi
done
echo "Process has been killed after $count seconds."
}

module load cuda/9.0
module load cudnn/7-cuda-9.0
module load openmpi/2.1.1-cuda9

export EXP_PATH=/scratch/$USER/ire
export NUM=1

mkdir -p $EXP_PATH
rm -rf $EXP_PATH/*
cp -r * $EXP_PATH
cd $EXP_PATH
mkdir -p logs/

# Pre process the data
python make_datafiles.py tweet_dataset > ./logs/processed

# Start training the model without coverage
echo "Starting Training without coverage"
python pointer-generator/run_summarization.py --mode=train --data_path=processed/chunked/train_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire &> logs/train_log &

sleep 1800

# Start evaluating models concurrently with training
echo "Starting Evaluation"
python pointer-generator/run_summarization.py --mode=eval --data_path=processed/chunked/val_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire &> logs/val_log &

# Train the model for a day
sleep 84600

# Kill the 2 processes above so that we can train with coverage now
pgrep "python" | while read line; do kill_process $line; done

# Copy the files so that we don't lose the model
mkdir -p ada:/share2/crvineeth97/ire/$NUM
rm -rf ada:/share2/crvineeth97/ire/$NUM/*
rsync -aP $EXP_PATH/* ada:/share2/crvineeth97/ire/$NUM
echo "Files copied successfully"

# Restore the best model we got after validation
python pointer-generator/run_summarization.py --mode=train --data_path=processed/chunked/train_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire --restore_best_model=1
echo "Best model restored"

# Again copy the files
rsync -aP $EXP_PATH/* ada:/share2/crvineeth97/ire/$NUM
echo "Copied with best model"

# Convert that best model into a coverage model so that we can train with coverage
echo "Training with coverage"
python pointer-generator/run_summarization.py --mode=train --convert_to_coverage_model=True --coverage=True --data_path=processed/chunked/train_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire &> logs/coverage_train_log &

sleep 1800

# Run validation concurrently
echo "Validating coverage"
python pointer-generator/run_summarization.py --mode=eval --coverage=True --data_path=processed/chunked/val_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire &> logs/coverage_val_log &

# Train for another hour and a half
sleep 5400

# Kill the 2 processes above so that we can finally test the model
pgrep "python" | while read line; do kill_process $line; done

# Copy the files
rsync -aP $EXP_PATH/* ada:/share2/crvineeth97/ire/$NUM
echo "Copied with coverage"

# Restore the best model we got after validation
python pointer-generator/run_summarization.py --mode=train --data_path=processed/chunked/train_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire --restore_best_model=1
echo "Best model with coverage restored"

# Copy files
rsync -aP $EXP_PATH/* ada:/share2/crvineeth97/ire/$NUM

# Run a single pass on the test data set to get the resultant summaries
python pointer-generator/run_summarization.py --mode=decode --single_pass=1 --data_path=processed/chunked/test_* --vocab_path=processed/vocab --log_root=./logs --exp_name=ire &> ./logs/generated
echo "Summaries generated"

# Finally copy everything again
rsync -aP $EXP_PATH/* ada:/share2/crvineeth97/ire/$NUM
