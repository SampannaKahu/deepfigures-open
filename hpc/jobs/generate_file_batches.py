import random
import json

if __name__ == "__main__":
    with open("crude_list.txt") as f:
        files = ["s3://" + file.strip().replace("_", "/", 2) for file in f if file and file.strip()]
    random.shuffle(files, random=lambda: random.uniform(0, 1))
    CHUNK_SIZE = 1
    final = [files[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] for i in range((len(files) + CHUNK_SIZE - 1) // CHUNK_SIZE)]
    for idx, final_list in enumerate(final):
        with open("../files_random_2633/files_{}.json".format(idx), mode="w") as file_json:
            json.dump(final_list, file_json, indent=4)
