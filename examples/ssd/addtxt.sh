#!
files=$0
for file in $files
do
echo file
temp="$file""~"
echo "import _init_paths" | cat - file0 > file1 ; # mv file1 file0
done
