def merge_files(file1_path, file2_path, file3_path, output_path):
    with open(file1_path, 'r') as file1:
        data1 = file1.read()

    with open(file2_path, 'r') as file2:
        data2 = file2.read()

    with open(file3_path, 'r') as file3:
        data3 = file3.read()

    merged_data = data1 + "\n" + data2 + "\n" + data3

    with open(output_path, 'w') as output_file:
        output_file.write(merged_data)

if __name__ == "__main__":
    file1_path = "/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/test.MGC"
    file2_path = "/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/train.MGC"
    file3_path = "/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/valid.MGC"
    output_path = "/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/data_analysis/MGC"

    merge_files(file1_path, file2_path, file3_path, output_path)
