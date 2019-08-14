#!/bin/bash
#SBATCH -N 5
#SBATCH -c 5
#SBATCH -g 2
#SBATCH --mem-per-cpu=12G
#SBATCH --time=0-24:00:00     # 30 minutes
#SBATCH --output=my.stdout
#SBATCH --mail-user=poulamirulz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name="just_a_test"

# Put commands for executing job below this line
# This example is loading the default Python module and then
# writing out the version of Python
module load singularity/2.5.1
for month  in /mnt/rds/redhen/gallina/tv/2019/*; do 
	for day in $month'/'*;do 
		for f in $(ls -R $day | grep DD-News.*.mp4);do 
			date +"%T"
			[ ! -f outputs/${f/mp4/ocr} ]  && echo $f $day'/' && singularity exec -B /mnt/rds/redhen/gallina/tv:/tv,`pwd`:/mnt ben-hin-ocr.img python3 /mnt/textdetection.py $f $day'/' 'hin+eng'
			date +"%T"
		done
	done 
done
echo "running OCR"
#singularity exec -B /mnt/rds/redhen/gallina/tv:/tv,`pwd`:/mnt ben-hin-ocr.img python3 /mnt/ticker.py
python --version
