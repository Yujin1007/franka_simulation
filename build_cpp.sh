#!/bin/bash

build_folder="build"
destination_folder="py_src/assets/cpp_library"
file_to_copy="controller.cpython-38-x86_64-linux-gnu.so"

if [ -d "$build_folder" ]; then
    echo "Deleting $build_folder..."
    rm -rf "$build_folder"
    echo "$build_folder deleted successfully."
fi

mkdir "$build_folder"
cd "$build_folder" || exit

echo -e "\nNow Running: cmake .."
cmake ..

echo -e "\nNow Running: make"
make

cd ..

if [ -e "$destination_folder/$file_to_copy" ]; then
    echo -e "\nRemoving existing $destination_folder/$file_to_copy..."
    rm "$destination_folder/$file_to_copy"
    echo "Existing file removed."
fi

new_filename="controller.so"
first_10_chars=$(echo "$file_to_copy" | cut -c 1-10)
if [ "$first_10_chars" == "controller" ]; then
    mv "$build_folder/$file_to_copy" "$build_folder/$new_filename"
    echo "$file_to_copy renamed successfully to $new_filename"
fi

echo -e "\nCopying $new_filename to $destination_folder..."
cp "$build_folder/$new_filename" "$destination_folder"
echo "$new_filename copied successfully."

echo -e "\nBuild process completed."