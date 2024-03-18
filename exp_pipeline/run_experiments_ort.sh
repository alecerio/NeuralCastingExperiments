INPUT_SIZE=(850 900 950 1000)
HIDDEN_SIZE=(850 900 950 1000)

bash_file_dir=$(dirname "$(realpath "$0")")
onnx_files_dir=$bash_file_dir/onnx_files
compiled_code_dir=$bash_file_dir/compile_nn_ort
num_experiments=10000

# clean onnx files dir
echo "clean onnx files directory ..."
rm $onnx_files_dir/*

# generate onnx files
for in in "${INPUT_SIZE[@]}"; do
    for h in "${HIDDEN_SIZE[@]}"; do
        echo "generate onnx file with input size $in and hidden size $h ..."
        python $bash_file_dir/pytorch_model/nsnet_pytorch_model_creator.py $in $h $onnx_files_dir
    done
done

onnx_file_list=$(ls "$onnx_files_dir")    
for file in $onnx_file_list; do
    # extract input and hidden size information
    if [[ $file =~ nsnet_([0-9]+)_([0-9]+).onnx ]]; then
        curr_input_size="${BASH_REMATCH[1]}"
        curr_hidden_size="${BASH_REMATCH[2]}"
        echo "input size: $curr_input_size"
        echo "hidden number: $curr_hidden_size"
    else
        echo "Invalid file name format"
    fi 
    
    # run executable
    echo "run executable nsnet with input size $curr_input_size and hidden size $curr_hidden_size ..."
    $compiled_code_dir/nsnet $onnx_files_dir/$file $curr_input_size $curr_hidden_size $compiled_code_dir/results/output_ort_${curr_input_size}_${curr_hidden_size}.txt

    # clean trash
    echo "clean trash ..."
    trash-empty
done