branch=`git branch --show-current | tr -d [:space:]`
echo ${branch}
PYTHONPROFILEIMPORTTIME=1 python -c "import daisypy.optim" 2> ${branch}_import_time.txt
cat ${branch}_import_time.txt | tr -d " " | cut -d ":" -f 2 | tr "|" "," > ${branch}_import_time.csv
python summarize_import_time.py ${branch}_import_time.csv > ${branch}_import_time_summary.txt
cat ${branch}_import_time_summary.txt
