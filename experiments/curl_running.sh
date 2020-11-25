
echo "Processing method mobilev2 VOT2018 acc30frames 5e-1"

sh comparison_study_oim.sh mobilev2_l234 CURL VOT2018 ep 5e-1 1

echo "Processing method alex VOT2018 acc30frames 5e-1"

sh comparison_study_oim.sh alex CURL VOT2018 ep 5e-1 1

echo "Processing method r50 VOT2018 acc30frames 5e-1"

sh comparison_study_oim.sh r50_l234 CURL VOT2018 ep 5e-1 1

echo "Processing method alex UAV123 acc30frames 5e-1"

sh comparison_study_oim.sh alex CURL UAV123 ep 5e-1 1

echo "Processing method alex LaSOT acc30frames 5e-1"

sh comparison_study_oim.sh alex CURL LaSOT ep 5e-1 1