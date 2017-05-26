#!/bin/sh
content=$(pwd)/README-old.md
new_readme=$(pwd)/README.md

# Entering subdirectory
basepath=$(pwd)
mkdir -p pdfs
pushd pdfs > /dev/null

IFS=$'\n'
initial_stack=$(dirs -v | wc -l)
rm -f $new_readme
while read line; do
  section=$(echo $line | sed -n -r 's/^(#+)\s*([0-9.]+)\s*(.*)$/\1\t\2 \3/p')
  if [ ! -z "$section" ]; then
    while (( $(dirs -v | wc -l) - $initial_stack > $(echo $section | cut -f 1 | wc -m) - 2)); do
      popd > /dev/null
    done
    dir=$(echo $section | cut -f 2 | sed -r 's/\//\&/g')
    echo "Entering $dir"
    mkdir -p "$dir"
    pushd "$dir" > /dev/null
    echo "$line" >> $new_readme
    continue
  fi
  paper=$(echo $line | sed -n -r 's/^\*\*\[([0-9]+)\]\*\*.*\*\*(.*)\*\*.*\[\[pdf\]\]\(([^)]*)\).*$/\1 \2\t\3/p')
  filename=$(echo $paper | cut -f 1).pdf
  if [ ! -z "$paper" ]; then
    if [ ! -f "$filename" ]; then
      echo "Donwloading $filename"
      curl $(echo $paper | cut -f 2) -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' -o "$filename" -L -s
      #if [ $(file "$filename"  --mime-type -b) != application/pdf ]; then
      #  rm "$filename"
      #fi
    fi
    path=$(realpath "--relative-to=$basepath" "$(pwd)/$filename")
    path=$(perl -MURI::Escape -e 'print uri_escape($ARGV[0]);' "$path")
    path=$(echo $path | sed 's/%2F/\\\//g')
    echo "$(echo $line | sed -r "s/^(.*\[\[pdf\]\]\()[^)]*(\).*)\$/\1$path\2/")" >> $new_readme
    continue
  fi
  echo "$line" >> $new_readme
done < $content
